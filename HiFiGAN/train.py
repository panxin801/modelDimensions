import warnings
warnings.filterwarnings("ignore")

import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import time
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from itertools import chain


import utils
import meldataset
from models import (Generator, MultiPeriodDiscriminator,
                    MultiScaleDiscriminator, discriminator_loss, generator_loss, feature_loss)

torch.backends.cudnn.benchmark = True


def train(rank, configs, logger):
    dist.init_process_group(
        backend=configs.dist_config.dist_backend,
        init_method=configs.dist_config.dist_url,
        world_size=configs.dist_config.world_size * configs.num_gpus,
        rank=rank)

    torch.cuda.set_device(rank)

    generator = Generator(configs).cuda(rank)
    mpd = MultiPeriodDiscriminator().cuda(rank)
    msd = MultiScaleDiscriminator().cuda(rank)

    if rank == 0:
        logger.info(generator)
        os.makedirs(configs.checkpoint_path, exist_ok=True)
        print(f"Checkpoints directory: {configs.checkpoint_path}")
        # save to yaml file
        with open(os.path.join(configs.checkpoint_path, "config.yaml"), "wt", encoding="utf8") as fw:
            OmegaConf.save(configs, fw)

    # Scan ckpts
    if os.path.isdir(configs.checkpoint_path):
        cp_g = utils.scan_checkpoint(configs.checkpoint_path, "g_")
        cp_do = utils.scan_checkpoint(configs.checkpoint_path, "do_")

    # Recover from checkpoint
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = utils.load_checkpoint(cp_g)
        state_dict_do = utils.load_checkpoint(cp_do)

        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        msd.load_state_dict(state_dict_do["msd"])

        steps = state_dict_g["steps"] + 1
        last_epoch = state_dict_do["epoch"]

    if configs.num_gpus > 1:
        generator = DDP(generator, device_ids=[rank])
        mpd = DDP(mpd, device_ids=[rank])
        msd = DDP(msd, device_ids=[rank])

    # Define optimizers
    optim_g = optim.AdamW(generator.parameters(), configs.learning_rate, betas=[
                          configs.adam_b1, configs.adam_b2])
    optim_do = optim.AdamW(chain(mpd.parameters(), msd.parameters()),
                           configs.learning_rate, betas=[configs.adam_b1, configs.adam_b2])

    if not state_dict_do is None:
        optim_do.load_state_dict(state_dict_do["optim_do"])
        optim_g.load_state_dict(state_dict_g["optim_g"])

    # Define learning_rate schedulers
    scheduler_g = optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=configs.lr_decay, last_epoch=last_epoch)
    scheduler_do = optim.lr_scheduler.ExponentialLR(
        optim_do, gamma=configs.lr_decay, last_epoch=last_epoch)

    # Get train and validate filelist
    training_filelist, validation_filelist = meldataset.get_dataset_filelist(
        configs)
    # Get  train dataset
    trainset = meldataset.MelDataset(training_filelist,
                                     configs.segment_size,
                                     configs.n_fft,
                                     configs.num_mels,
                                     configs.hop_size,
                                     configs.win_size,
                                     configs.sampling_rate,
                                     configs.fmin,
                                     configs.fmax,
                                     n_cache_reuse=0,
                                     shuffle=False if configs.num_gpus > 1 else True,
                                     fmax_loss=configs.fmax_for_loss,
                                     device=None,
                                     fine_tuning=configs.fine_tuning,
                                     base_mels_path=configs.input_mels_dir)
    train_sampler = data.DistributedSampler(
        trainset) if configs.num_gpus > 1 else None
    train_loader = data.DataLoader(trainset, num_workers=configs.num_workers,
                                   shuffle=False,
                                   batch_size=configs.batch_size,
                                   sampler=train_sampler,
                                   pin_memory=True,
                                   drop_last=True
                                   )
    # Define validation dataset
    if rank == 0:
        validset = meldataset.MelDataset(validation_filelist, configs.segment_size,
                                         configs.n_fft,
                                         configs.num_mels,
                                         configs.hop_size,
                                         configs.win_size,
                                         configs.sampling_rate,
                                         configs.fmin,
                                         configs.fmax,
                                         False,
                                         False,
                                         n_cache_reuse=0,
                                         fmax_loss=configs.fmax_for_loss,
                                         device=None,
                                         fine_tuning=configs.fine_tuning,
                                         base_mels_path=configs.input_mels_dir)
        validation_loader = data.DataLoader(validset,
                                            num_workers=1,
                                            shuffle=False,
                                            batch_size=1,
                                            sampler=None,
                                            pin_memory=True,
                                            drop_last=True
                                            )
        sw = SummaryWriter(os.path.join(configs.checkpoint_path, "logs"))

    # Prepare to train
    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), configs.training_epochs):
        if rank == 0:
            start = time.time()
            logger.info(f"Epoch: {epoch+1}")

        if configs.num_gpus > 1:  # Shuffle sampler for distributed training
            train_loader.sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()

            x, y, _, y_mel = batch
            x = torch.autograd.Variable(x.cuda(rank, non_blocking=True))
            y = torch.autograd.Variable(y.cuda(rank, non_blocking=True))
            y_mel = torch.autograd.Variable(
                y_mel.cuda(rank, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
            y_g_hat_mel = meldataset.mel_spectrogram(
                y_g_hat.squeeze(1), configs.n_fft, configs.num_mels, configs.sampling_rate, configs.hop_size, configs.win_size, configs.fmin, configs.fmax_for_loss)

            optim_do.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(
                y, y_g_hat.detach())  # mpd 不影响generator。
            loss_disc_f, loss_disc_f_r, loss_disc_f_g = discriminator_loss(
                y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_f + loss_disc_s
            loss_disc_all.backward()
            optim_do.step()

            # Generator
            optim_g.zero_grad()
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % configs.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print("Steps: {: d}, Gen Loss Total: {: 4.3f}, Mel - Spec. Error: {: 4.3f}, s / b: {: 4.3f}".
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))

                # checkpointing
                if steps % configs.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(
                        configs.checkpoint_path, steps)
                    utils.save_checkpoint(checkpoint_path,
                                          {"generator": (generator.module if configs.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(
                        configs.checkpoint_path, steps)
                    utils.save_checkpoint(checkpoint_path,
                                          {"mpd": (mpd.module if configs.num_gpus > 1
                                                   else mpd).state_dict(),
                                           "msd": (msd.module if configs.num_gpus > 1
                                                   else msd).state_dict(),
                                           "optim_g": optim_g.state_dict(), "optim_d": optim_do.state_dict(), "steps": steps,
                                           "epoch": epoch})

                # Tensorboard summary logging
                if steps % configs.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total",
                                  loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % configs.validation_interval == 0:  # and steps !=0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0

                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            y_g_hat = generator(x.cuda(rank))
                            y_mel = torch.autograd.Variable(
                                y_mel.cuda(rank, non_blocking=True))
                            y_g_hat_mel = meldataset.mel_spectrogram(y_g_hat.squeeze(1), configs.n_fft, configs.num_mels, configs.sampling_rate,
                                                                     configs.hop_size, configs.win_size,
                                                                     configs.fmin, configs.fmax_for_loss)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio(
                                        "gt/y_{}".format(j), y[0], steps, configs.sampling_rate)
                                    sw.add_figure(
                                        "gt/y_spec_{}".format(j), utils.plot_spectrogram(x[0]), steps)

                                sw.add_audio(
                                    "generated/y_hat_{}".format(j), y_g_hat[0], steps, configs.sampling_rate)
                                y_hat_spec = meldataset.mel_spectrogram(y_g_hat.squeeze(1), configs.n_fft, configs.num_mels,
                                                                        configs.sampling_rate, configs.hop_size, configs.win_size,
                                                                        configs.fmin, configs.fmax)
                                sw.add_figure("generated/y_hat_spec_{}".format(j),
                                              utils.plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar(
                            "validation/mel_spec_error", val_err, steps)

                    generator.train()
                steps += 1
            scheduler_g.step()
            scheduler_do.step()

            if rank == 0:
                print("Time taken for epoch {} is {} sec\n".format(
                    epoch + 1, int(time.time() - start)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_name", default=None)
    parser.add_argument("--config", default="./config_v1.yaml")
    parse = parser.parse_args()

    configs = OmegaConf.load(parse.config)
    logger = utils.getLogger(configs.modelDir)

    # Set seed
    torch.manual_seed(configs.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.seed)
        configs.num_gpus = torch.cuda.device_count()
        configs.batch_size = int(configs.batch_size / configs.num_gpus)
        logger.info(f"Batch size per GPU: {configs.batch_size}")
    else:
        pass

    # Distributed training if more than 1 GPU is available
    # mp.spawn(train, nprocs=configs.num_gpus, args=(configs, logger ))
    train(0, configs, logger)


if __name__ == "__main__":
    main()
