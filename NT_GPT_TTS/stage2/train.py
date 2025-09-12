import argparse
import numpy as np
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import json
import warnings
from scipy.io.wavfile import write
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import vits.commons as commons
import vits.utils as viutils

from utils import (Hparams)
from dataset import (DistributedBucketSampler,
                     TokenVocoderCollate, TokenVocoderDataset)
from stage2 import (SynthesizerTrn, MultiPeriodDiscriminator)
from vits.losses import (
    generator_loss, discriminator_loss, feature_loss, kl_loss)
from vits.mel_processing import (mel_spectrogram_torch, spec_to_mel_torch)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

global_step = 0


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--meta-file", type=str, default="total.txt")
    parser.add_argument("--token-scp", type=str, default="token.scp")
    parser.add_argument("--mel-scp", type=str, default="mel16k.scp")
    parser.add_argument("--model-dir", type=str, default="stage2model")
    parser.add_argument("--result-dir", type=str, default="result_audio")
    parser.add_argument("--config-path", type=str, default=r"vctk_config.json")

    args = parser.parse_args()
    return args


def train(rank, args, hparams):
    os.makedirs(args.model_dir, exist_ok=True)

    # init dist training
    if args.num_gpus > 1:
        print(f"Init distributed training on rank {rank}")

        os.environ["MASTER_ADDR"] = hparams.dist_url
        os.environ["MASTER_PORT"] = hparams.dist_port

        dist.init_process_group(backend=hparams.dist_backend,
                                init_method="env://",
                                world_size=hparams.dist_world_size * args.num_gpus,
                                rank=rank)
        print(f"Initialized process group for rank {rank} successfully")

    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.model_dir, "logs"))

    torch.cuda.set_device(rank)

    # Load data
    trainDataset = TokenVocoderDataset(args, hparams, "train")
    trainSampler = DistributedBucketSampler(trainDataset,
                                            hparams.batch_size,
                                            [32, 300, 400, 500, 600,
                                             700, 800, 900, 1000],
                                            num_replicas=args.num_gpus,
                                            rank=rank,
                                            shuffle=True)
    collateFn = TokenVocoderCollate(segmentSize=hparams.segment_size)
    trainLoader = torch.utils.data.DataLoader(trainDataset,
                                              shuffle=False,
                                              batch_sampler=trainSampler,
                                              collate_fn=collateFn,
                                              num_workers=8,
                                              pin_memory=True,
                                              drop_last=False)

    if rank == 0:
        evalDataset = TokenVocoderDataset(args, hparams, "valid")
        evalLoader = torch.utils.data.DataLoader(evalDataset,
                                                 shuffle=False,
                                                 batch_size=1,
                                                 num_workers=4,
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 collate_fn=collateFn)

    # Init model
    net_g = SynthesizerTrn(
        n_vocab=513,
        spec_channels=101,
        segment_size=100,
    ).cuda()
    net_d = MultiPeriodDiscriminator().cuda()
    optim_g = torch.optim.AdamW(net_g.parameters(),
                                hparams.learning_rate,
                                betas=hparams.betas,
                                eps=hparams.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hparams.learning_rate,
        betas=hparams.betas,
        eps=hparams.eps
    )
    if args.num_gpus > 1:
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])

    epoch_str = 1

    scheduler_g = optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hparams.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hparams.lr_decay, last_epoch=epoch_str - 2)

    scalar = torch.amp.GradScaler(enabled=hparams.use_fp16)

    # Train loop
    for epoch in range(epoch_str, hparams.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, args, hparams, [net_g, net_d], [optim_g, optim_d], [
                               scheduler_g, scheduler_d], scalar, [trainLoader, evalLoader], None, [writer, writer])
        else:
            train_and_evaluate(rank, epoch, args, hparams, [net_g, net_d], [optim_g, optim_d], [
                               scheduler_g, scheduler_d], scalar, [trainLoader, None], None, None)

        scheduler_d.step()
        scheduler_g.step()

    print(f"Train done!")


def train_and_evaluate(rank, epoch, args, hparams, nets, optims, schedulers, scalar, loaders, logger, writers):
    global global_step

    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders

    if not writer is None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)

    pbar = tqdm(train_loader)
    net_g.train()
    net_d.train()
    for batch_idx, batch in enumerate(pbar):
        token_padded, token_lens, mels, audio_padded, audio_lens, spec_padded, spec_lens = batch
        token_padded = token_padded.cuda().long()
        token_lens = token_lens.cuda().long()
        mels = mels.cuda().float()
        audio_padded = audio_padded.cuda().float()
        audio_lens = audio_lens.cuda().long()
        spec_padded = spec_padded.cuda().float()
        spec_lens = spec_lens.cuda().long()

        with torch.amp.autocast(enabled=hparams.use_fp16, dtype=torch.bfloat16, device_type="cuda"):
            y_hat, token_length, ids_slice, x_mask, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                    token_padded, token_lens, spec_padded, spec_lens, mels)
            y_hat = y_hat.squeeze(1)
            mel = spec_to_mel_torch(
                spec_padded,
                hparams.filter_length,
                hparams.n_mel_channels,
                hparams.sampling_rate,
                hparams.mel_fmin,
                hparams.mel_fmax
            )

            y_mel = commons.slice_segments(
                mel, ids_slice, hparams.segment_size // hparams.hop_length)
            result_min_size = min(y_hat.size(1), audio_padded.size(1))

            y_hat = y_hat[:, :result_min_size]

            audio_padded = audio_padded.unsqueeze(1)
            y_hat_mel = mel_spectrogram_torch(
                y_hat,
                hparams.filter_length,
                hparams.n_mel_channels,
                hparams.sampling_rate,
                hparams.hop_length,
                hparams.win_length,
                hparams.mel_fmin,
                hparams.mel_fmax
            )
            y_hat = y_hat.unsqueeze(1)
            audio_padded = commons.slice_segments(
                audio_padded, ids_slice * hparams.hop_length, hparams.segment_size)

            y_d_hat_r, y_d_hat_g, _, _ = net_d(audio_padded, y_hat.detach())

            with torch.amp.autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

        optim_d.zero_grad()
        scalar.scale(loss_disc_all).backward()
        scalar.unscale_(optim_d)

        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scalar.step(optim_d)

        with torch.amp.autocast(enabled=hparams.fp16_run, dtype=torch.bfloat16, device_type="cuda"):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(audio_padded, y_hat)
            with torch.amp.autocast(enabled=False):
                min_len = min(logs_p.shape[2], logs_q.shape[2])
                z_p = z_p[:, :, :min_len]
                logs_q = logs_q[:, :, :min_len]
                m_p = m_p[:, :, :min_len]
                logs_p = logs_p[:, :, :min_len]
                z_mask = z_mask[:, :, :min_len]

                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hparams.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p,
                                  z_mask) * hparams.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)

                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        optim_g.zero_grad()
        scalar.scale(loss_gen_all).backward()
        scalar.unscale_(optim_g)

        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)

        scalar.step(optim_g)
        scalar.update()

        if rank == 0:
            if global_step % hparams.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                print('Train Epoch: {}[{:.0f}%]'.format(
                    epoch, 100. * batch_idx / len(train_loader)))
                print([x.item() for x in losses] + [global_step, lr])
                scalar_dict = {'loss/g/total': loss_gen_all, 'loss/d/total': loss_disc_all,
                               'learning_rate': lr, 'grad_norm_d': grad_norm_d, 'grad_norm_g': grad_norm_g}
                scalar_dict.update(
                    {'loss/g/fm': loss_fm, 'loss/g/mel': loss_mel, 'loss/g/kl': loss_kl, })
                scalar_dict.update(
                    {'loss/g/{}'.format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update(
                    {'loss/d_r/{}'.format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update(
                    {'loss/d_g/{}'.format(i): v for i, v in enumerate(losses_disc_g)})

                viutils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict
                )

                if global_step % hparams.eval_interval == 0:
                    evaluate(rank, args, hparams, net_g,
                             eval_loader, writer_eval)
                    viutils.save_checkpoint(net_g, optim_g, hparams.learning_rate, epoch, os.path.join(
                        args.model_dir, 'G_{}.pth'.format(global_step)))
                    viutils.save_checkpoint(net_d, optim_d, hparams.learning_rate, epoch, os.path.join(
                        args.model_dir, 'D_{}.pth'.format(global_step)))

        global_step += 1
    if rank == 0:
        print('======> Epoch: {}'.format(epoch))


def evaluate(rank, args, hparams, generator, eval_loader, writer_eval):
    global global_step
    nums = 0
    generator.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(eval_loader):
            token_padded, token_lens, mels, audio_padded, audio_lens, spec_padded, spec_lens = batch

            token_padded = token_padded.cuda().long()
            token_lens = token_lens.cuda().long()
            mels = mels.cuda().float()
            audio_padded = audio_padded.cuda().float()
            audio_lens = audio_lens.cuda().long()
            spec_padded = spec_padded.cuda().float()
            spec_lens = spec_lens.cuda().long()

            if args.num_gpus > 1:
                y_hat, mask = generator.module.infer(
                    token_padded, token_lens, mels, max_len=int(audio_lens[0]))
            else:
                y_hat, mask = generator.infer(
                    token_padded, token_lens, mels, max_len=int(audio_lens[0]))

            y_hat = y_hat.cpu().data.numpy()
            os.makedirs(os.path.join(args.result_dir,
                        str(global_step)), exist_ok=True)
            out_path = os.path.join(args.result_dir, str(
                global_step), str(batch_idx) + '.wav')

            write(out_path, hparams.sampling_rate, y_hat)
            nums += 1
            if nums == 20:
                break
    generator.train()


if __name__ == "__main__":
    args = get_args()

    with open(args.config_path, "rt", encoding="utf8") as fr:
        data = fr.read()
    config = json.loads(data)
    hparams = Hparams(**config)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    torch.backends.cudnn.cudnn_deterministic = hparams.cudnn_deterministic

    if torch.cuda.is_available():
        torch.cuda.manual_seed(hparams.seed)
        # torch.cuda.manual_seed_all(hparams.seed)
        torch.manual_seed(hparams.seed)

        args.num_gpus = torch.cuda.device_count()
        args.batch_size = int(hparams.batch_size / args.num_gpus)
        print(
            f"Using {args.num_gpus} GPUs, batch size per GPU: {args.batch_size}")
    else:
        print(f"Using CPU, batch size: {hparams.batch_size}")

    if args.num_gpus > 1:
        mp.spawn(train, nprocs=args.num_gpus, args=(args, hparams))
    else:
        train(0, args, hparams)
