import logging
import argparse
import datetime
import platform
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as dataC
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import (autocast, GradScaler)
from tqdm import tqdm

logging.getLogger("numba").setLevel(logging.WARNING)
import utils
from config import config
from data_utils import (TextAudioSpeakerLoader,
                        TextAudioSpeakerCollate,
                        DistributedBucketSampler,)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
    WavLMDiscriminator,
)
from text.symbols import symbols
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
    WavLMLoss,
)

torch.backends.cuda.matmul.allow_tf32 = True
# If encountered training problem,please try to disable TF32.
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
# Not available if torch version is lower than 2.0
torch.backends.cuda.enable_mem_efficient_sdp(True)

global_step = 0


def run():
    # 环境变量解析
    envs = config.train_ms_config.env
    for env_name, env_value in envs.items():
        print("加载config中的配置{}".format(str(env_value)))
        os.environ[env_name] = str(env_value)
    print(
        "加载环境变量 \nMASTER_ADDR: {},\nMASTER_PORT: {},\nWORLD_SIZE: {},\nRANK: {},\nLOCAL_RANK: {}".format(
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
            os.environ["WORLD_SIZE"],
            os.environ["RANK"],
            os.environ["LOCAL_RANK"],
        )
    )

    backend = "nccl"
    if platform.system() == "Windows":
        backend = "gloo"  # If Windows,switch to gloo backend.

    dist.init_process_group(backend,
                            init_method="env://",
                            timeout=datetime.timedelta(seconds=120),
                            )  # Use torchrun instead of mp.spawn
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    n_gpus = dist.get_world_size()

    # 命令行/config.yml配置解析
    # hps = utils.get_hparams()
    parser = argparse.ArgumentParser()
    # 非必要不建议使用命令行配置，请使用config.yml文件
    parser.add_argument("-c", "--config",
                        type=str,
                        default=config.train_ms_config.config_path,
                        help="JSON file for configuration"
                        )
    parser.add_argument("-m", "--model",
                        type=str,
                        default=config.dataset_path,
                        help="数据集文件夹路径，请注意，数据不再默认放在/logs文件夹下。如果需要用命令行配置，请声明相对于根目录的路径",
                        )
    args, _ = parser.parse_known_args()

    model_dir = os.path.join(args.model, config.train_ms_config.model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.config)
    hps.model_dir = model_dir
    # 比较路径是否相同
    if os.path.realpath(args.config) != os.path.realpath(config.train_ms_config.config_path):
        with open(args.config, "rt", encoding="utf8") as f:
            data = f.read()
        with open(config.train_ms_config.config_path, "wt", encoding="utf8") as f:
            f.write(data)

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(local_rank)

    global global_step

    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(
            log_dir=os.path.join(hps.model_dir, "eval"))

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate()
    train_loader = dataC.DataLoader(
        train_dataset,
        num_workers=min(config.train_ms_config.num_workers,
                        os.cpu_count() - 2),
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=4,
    )  # DataLoader config could be adjusted.
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(
            hps.data.validation_files, hps.data)
        eval_loader = dataC.DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    if ("use_noise_scaled_mas" in hps.model.keys()
            and hps.model.use_noise_scaled_mas is True):
        print("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0

    if (
        "use_duration_discriminator" in hps.model.keys()
        and hps.model.use_duration_discriminator is True
    ):
        print("Using duration discriminator for VITS2")
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(local_rank)
    else:
        net_dur_disc = None

    if (
        "use_spk_conditioned_encoder" in hps.model.keys()
        and hps.model.use_spk_conditioned_encoder is True
    ):
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    else:
        print("Using normal encoder for VITS1")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).cuda(local_rank)

    if getattr(hps.train, "freeze_ZH_bert", False):
        print("Freezing ZH bert encoder !!!")
        for param in net_g.enc_p.bert_proj.parameters():
            param.requires_grad = False

    if getattr(hps.train, "freeze_EN_bert", False):
        print("Freezing EN bert encoder !!!")
        for param in net_g.enc_p.en_bert_proj.parameters():
            param.requires_grad = False

    if getattr(hps.train, "freeze_JP_bert", False):
        print("Freezing JP bert encoder !!!")
        for param in net_g.enc_p.ja_bert_proj.parameters():
            param.requires_grad = False

    net_d = MultiPeriodDiscriminator(
        hps.train.use_spectral_norm).cuda(local_rank)
    net_wd = WavLMDiscriminator(
        hps.model.slm.hidden, hps.model.slm.nlayers, hps.model.slm.initial_channel).cuda(local_rank)

    optim_g = optim.AdamW(filter(lambda p: p.requires_grad, net_g.parameters()),
                          hps.train.learning_rate,
                          betas=hps.train.betas,
                          eps=hps.train.eps)
    optim_d = optim.AdamW(net_d.parameters(),
                          hps.train.learning_rate,
                          betas=hps.train.betas,
                          eps=hps.train.eps)
    optim_wd = optim.AdamW(net_wd.parameters(),
                           hps.train.learning_rate,
                           betas=hps.train.betas,
                           eps=hps.train.eps)
    if not net_dur_disc is None:
        optim_dur_disc = optim.AdamW(net_dur_disc.parameters(),
                                     hps.train.learning_rate,
                                     betas=hps.train.betas,
                                     eps=hps.train.eps)
    else:
        optim_dur_disc = None

    net_g = DDP(net_g, device_ids=[local_rank], bucket_cap_mb=512)
    net_d = DDP(net_d, device_ids=[local_rank], bucket_cap_mb=512)
    net_wd = DDP(net_wd, device_ids=[local_rank], bucket_cap_mb=512)
    if not net_dur_disc is None:
        net_dur_disc = DDP(net_dur_disc, device_ids=[
                           local_rank], bucket_cap_mb=512)

    # 下载底模
    if config.train_ms_config.base["use_base_model"]:
        utils.download_checkpoint(
            hps.model_dir,
            config.train_ms_config.base,
            token=config.openi_token,
            mirror=config.mirror,
        )
    dur_resume_lr = hps.train.learning_rate
    wd_resume_lr = hps.train.learning_rate
    if not net_dur_disc is None:
        try:
            _, _, dur_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                net_dur_disc,
                optim_dur_disc,
                skip_optimizer=(
                    hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
                ),
            )
            if not optim_dur_disc.param_groups[0].get("initial_lr"):
                optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
        except:
            print("Initialize dur_disc")

    try:
        _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
            net_g,
            optim_g,
            skip_optimizer=(
                hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
            ),
        )
        _, optim_d, d_resume_lr, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
            net_d,
            optim_d,
            skip_optimizer=(
                hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
            ),
        )
        if not optim_g.param_groups[0].get("initial_lr"):
            optim_g.param_groups[0]["initial_lr"] = g_resume_lr
        if not optim_d.param_groups[0].get("initial_lr"):
            optim_d.param_groups[0]["initial_lr"] = d_resume_lr

        epoch_str = max(epoch_str, 1)
        # global_step = (epoch_str - 1) * len(train_loader)
        global_step = int(
            utils.get_steps(utils.latest_checkpoint_path(
                hps.model_dir, "G_*.pth"))
        )
        print(
            f"******************检测到模型存在，epoch为 {epoch_str}，gloabl step为 {global_step}*********************"
        )
    except Exception as e:
        print(e)
        epoch_str = 1
        global_step = 0

    try:
        _, optim_wd, wd_resume_lr, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "WD_*.pth"),
            net_wd,
            optim_wd,
            skip_optimizer=(
                hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
            ),
        )
        if not optim_wd.param_groups[0].get("initial_lr"):
            optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr
    except Exception as e:
        print(e)

    scheduler_g = optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_wd = optim.lr_scheduler.ExponentialLR(
        optim_wd, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        scheduler_dur_disc = None

    scaler = GradScaler(enabled=hps.train.bf16_run)

    wl = WavLMLoss(
        hps.model.slm.model,
        net_wd,
        hps.data.sampling_rate,
        hps.model.slm.sr
    ).cuda(local_rank)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                local_rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc, net_wd, wl],
                [optim_g, optim_d, optim_dur_disc, optim_wd],
                [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
            )
        else:
            train_and_evaluate(
                rank,
                local_rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc, net_wd, wl],
                [optim_g, optim_d, optim_dur_disc, optim_wd],
                [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
                scaler,
                [train_loader, None],
                None,
                None,
            )
        scheduler_g.step()
        scheduler_d.step()
        scheduler_wd.step()
        if not net_dur_disc is None:
            scheduler_dur_disc.step()


def train_and_evaluate(
    rank,
    local_rank,
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    logger,
    writers,
):
    ...


if __name__ == "__main__":
    run()
