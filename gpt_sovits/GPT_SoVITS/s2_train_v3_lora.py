import warnings
warnings.filterwarnings("ignore")
import os
import sys
import torch
import logging
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data as Data
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.amp import (GradScaler, autocast)
from collections import OrderedDict
from peft import (LoraConfig, get_peft_model)

import utils
from module import commons
from module.data_utils import (
    DistributedBucketSampler,
    TextAudioSpeakerCollateV3,
    TextAudioSpeakerLoaderV3,
    TextAudioSpeakerCollateV4,
    TextAudioSpeakerLoaderV4,

)
from module.models import (
    SynthesizerTrnV3 as SynthesizerTrn,
)
from process_ckpt import savee


logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
# 反正A100fp32更快，那试试tf32吧
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")  # 最低精度但最快（也就快一丁点），对于结果造成不了影响
global_step = 0
device = "cpu"


def run(rank, n_gpus, hps):
    """ Train func
    """
    global global_step, no_grad_names, save_root, lora_rank
    if rank == 0:
        logger = utils.get_logger(hps.data.exp_dir)
        logger.info(hps)
        writer = SummaryWriter(log_dir=hps.s2_ckpt_dir)
        writer_eval = SummaryWriter(
            log_dir=os.path.join(hps.s2_ckpt_dir, "eval"))

    dist.init_process_group(backend="gloo" if os.name ==
                            "nt" or not torch.cuda.is_available() else "nccl",
                            init_method="env://?use_libuv=False",
                            world_size=n_gpus,
                            rank=rank)
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    TextAudioSpeakerLoader = TextAudioSpeakerLoaderV3 if hps.model.version == "v3" else TextAudioSpeakerLoaderV4
    TextAudioSpeakerCollate = TextAudioSpeakerCollateV3 if hps.model.version == "v3" else TextAudioSpeakerCollateV4
    train_dataset = TextAudioSpeakerLoader(hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [
            32,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            # 1100,
            # 1200,
            # 1300,
            # 1400,
            # 1500,
            # 1600,
            # 1700,
            # 1800,
            # 1900,
        ],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate()
    train_loader = Data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_sampler=train_sampler,
        num_workers=6,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    save_root = f"{hps.data.exp_dir}/logs_s2_{hps.model.version}_lora_{hps.train.lora_rank}"
    os.makedirs(save_root, exist_ok=True)
    lora_rank = int(hps.train.lora_rank)
    lora_config = LoraConfig(
        target_modules=["to_k", "to_q", "to_v",
                        "to_out.0"],  # 在DiT的Attention中，这些层用lora
        r=lora_rank,
        lora_alpha=lora_rank,  # 一般设置为2*lora_rank
        init_lora_weights=True,
    )  # 定义lora 配置

    def get_model(hps):
        return SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )

    def get_optim(net_g):
        return optim.AdamW(
            filter(lambda p: p.requires_grad, net_g.parameters()),  # 默认所有层lr一致
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )

    def model2cuda(net_g, rank):
        if torch.cuda.is_available():
            net_g = DDP(net_g.cuda(rank), device_ids=[
                        rank], find_unused_parameters=True)
        else:
            net_g.to(device)
        return net_g

    try:  # 如果能加载自动resume
        net_g = get_model(hps)
        # 得到带了lora配置的peft模型，很多模型参数的requires_grad都为False,可以用param.numel和requires_grad组合起来一起看。
        net_g.cfm = get_peft_model(net_g.cfm, lora_config)
        net_g = model2cuda(net_g, rank)
        optim_g = get_optim(net_g)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(save_root, "G_*.pth"),
            net_g,
            optim_g,
        )  # 这里可能会读取失败 就会进入重头开始训练
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
    except:  # 如果首次不能加载，加载pretrain
        epoch_str = 1
        global_step = 0
        net_g = get_model(hps)
        if (
            hps.train.pretrained_s2G != "" and
            not hps.train.pretrained_s2G is None and
            os.path.exists(hps.train.pretrained_s2G)
        ):
            if rank == 0:
                logger.info(f"loaded pretrained {hps.train.pretrained_s2G}")
            print(
                f"loaded pretrained {hps.train.pretrained_s2G}",
                net_g.load_state_dict(torch.load(hps.train.pretrained_s2G, map_location="cpu")["weight"],
                                      strict=False)
            )
        net_g.cfm = get_peft_model(net_g.cfm, lora_config)
        net_g = model2cuda(net_g, rank)
        optim_g = get_optim(net_g)

    no_grad_names = set()
    for name, param in net_g.named_parameters():
        if not param.requires_grad:
            no_grad_names.add(name.replace("module.", ""))

    scheduler_g = optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=-1)
    for _ in range(epoch_str):
        scheduler_g.step()

    scaler = GradScaler(enabled=hps.train.fp16_run)

    net_d = optim_d = scheduler_d = None
    print(f"start training from epoch {epoch_str}")
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
            )
        scheduler_g.step()
    print("Training Done!")


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if not writers is None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step
    net_g.train()
    for batch_idx, (ssl, spec, mel, ssl_lengths, spec_lengths, text, text_lengths, mel_lengths) in enumerate(tqdm(train_loader)):
        if torch.cuda.is_available():
            spec, spec_lengths = spec.cuda(
                rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
            mel, mel_lengths = mel.cuda(
                rank, non_blocking=True), mel_lengths.cuda(rank, non_blocking=True)
            # ssl.requires_grad 本身也是 False
            ssl = ssl.cuda(rank, non_blocking=True)
            ssl.requires_grad = False
            text, text_lengths = text.cuda(
                rank, non_blocking=True), text_lengths.cuda(rank, non_blocking=True)
        else:
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            mel, mel_lengths = mel.to(device), mel_lengths.to(device)
            ssl = ssl.to(device)
            ssl.requires_grad = False
            text, text_lengths = text.to(device), text_lengths.to(device)

        with autocast(device_type="cuda", enabled=hps.train.fp16_run):
            cfm_loss = net_g(
                ssl,
                spec,
                mel,
                ssl_lengths,
                spec_lengths,
                text,
                text_lengths,
                mel_lengths,
                use_grad_ckpt=hps.train.grad_ckpt,
            )
            loss_gen_all = cfm_loss

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [cfm_loss]
                logger.info(
                    f"Train Epoch: {epoch} [{100.0 * batch_idx/len(train_loader):.0f}%]")
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss_gen_all,
                               "learning_rate": lr, "grad_norm_g": grad_norm_g}
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict,
                )
        global_step += 1
    if epoch % hps.train.save_every_epoch == 0 and rank == 0:
        if hps.train.if_save_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(save_root, "G_{}.pth".format(global_step)),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(save_root, "G_{}.pth".format(233333333333)),
            )
        if rank == 0 and hps.train.if_save_every_weights == True:
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            sim_ckpt = OrderedDict()
            for key in ckpt:
                if not key in no_grad_names:
                    sim_ckpt[key] = ckpt[key].half().cpu()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        sim_ckpt,
                        hps.name +
                        "_e%s_s%s_l%s" % (epoch, global_step, lora_rank),
                        epoch,
                        global_step,
                        hps, cfm_version=hps.model.version,
                        lora_rank=lora_rank,
                    ),
                )
            )

    if rank == 0:
        logger.info(f"====> Epoch: {epoch}")


def main():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "62221"

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


if __name__ == "__main__":
    hps = utils.get_hparams(stage=2)
    os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace(
        "-", ",")

    main()
