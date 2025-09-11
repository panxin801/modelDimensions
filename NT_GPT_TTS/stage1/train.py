import argparse
import numpy as np
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import json
import warnings
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import (Hparams, count_parameters)
from dataset import (TransducerDataset, TransducerCollate)
from stage1 import Stage1Net
from optim import (Eden, Eve)


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def save_checkpoint(model, optimizer, lr, iteration, filepath):
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    dirName = os.path.dirname(filepath)
    os.makedirs(dirName, exist_ok=True)

    torch.save({
        "iteration": iteration,
        "state_dict": state_dict,
        "lr": lr,
        "optimizer": optimizer.state_dict()
    }, filepath)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--train-meta", type=str, default="total.txt")
    parser.add_argument("--valid-meta", type=str, default="total.txt")
    parser.add_argument("--mel-scp", type=str, default="mel16k.scp")
    parser.add_argument("--token-scp", type=str, default="token.scp")
    parser.add_argument("--load-iteration", type=bool, default=True)
    parser.add_argument("--output-directory", type=str, default="stage1model")
    parser.add_argument("--config-path", type=str, default="vctk_config.json")

    args = parser.parse_args()
    return args


def train(rank, args, hparams):
    os.makedirs(args.output_directory, exist_ok=True)

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

    torch.cuda.set_device(rank)

    # Load datasets
    trainDataset = TransducerDataset(args.mel_scp,
                                     args.token_scp,
                                     args.train_meta,
                                     "train",
                                     hparams.segment_size)
    validDataset = TransducerDataset(args.mel_scp,
                                     args.token_scp,
                                     args.valid_meta,
                                     "valid",
                                     hparams.segment_size)
    collateFn = TransducerCollate(hparams.segment_size)

    if args.num_gpus > 1:
        trainSampler = torch.utils.data.distributed.DistributedSampler(
            trainDataset)
        shuffle = False
    else:
        trainSampler = None
        shuffle = True

    trainLoader = torch.utils.data.DataLoader(trainDataset,
                                              batch_size=args.batch_size,
                                              shuffle=shuffle,
                                              sampler=trainSampler,
                                              collate_fn=collateFn,
                                              num_workers=8,
                                              pin_memory=True,
                                              drop_last=False)
    validLoader = torch.utils.data.DataLoader(validDataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              collate_fn=collateFn,
                                              num_workers=4,
                                              pin_memory=False,
                                              drop_last=False)

    # Init model
    model = Stage1Net(text_dim=384,
                      num_vocabs=513,
                      num_phonemes=512,
                      token_dim=256,
                      hid_token_dim=512,
                      inner_dim=513,
                      ref_dim=513,
                      use_fp16=hparams.use_fp16)
    pams = count_parameters(model)
    print(f"Number of parameters: {pams}")

    if args.num_gpus > 1:
        model = DDP(model.cuda(), device_ids=[
                    rank], find_unused_parameters=True)
    else:
        model = model.cuda()

    iteration = 0
    accu_iter = 0
    epoch_offset = 0
    init_lr = hparams.initial_lr

    # Restore from ckpt
    if not args.checkpoint_path is None:
        assert os.path.isfile(
            args.checkpoint_path), f"{args.checkpoint_path} not found"

        checkpoint = torch.load(args.checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Load stage1 ckpt successful")

        if args.load_iteration:
            init_lr = checkpoint["lr"]
            iteration = checkpoint["iteration"]
            iteration += 1
            epoch_offset = int(iteration / len(trainLoader))
        print(f"Restore successful, {iteration=}, {epoch_offset=}")

    # init optimizer
    optimizer = Eve(model.parameters(), lr=init_lr)
    lr_scheduler = Eden(optimizer, 5000, 6)

    if not args.checkpoint_path is None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    for param in optimizer.param_groups:
        param["initial_lr"] = init_lr  # param is a dict

    # Train loop
    scaler = torch.amp.GradScaler(enabled=hparams.use_fp16)
    if rank == 0:
        sw = SummaryWriter(os.path.join(args.output_directory, 'logs'))

    for epoch in range(epoch_offset, hparams.epoches):
        model.train()
        print(f"{epoch=}")

        if args.num_gpus > 1:
            trainSampler.set_epoch(epoch)

        epochLoss = 0.0
        batchNum = 0
        pbar = tqdm(trainLoader)
        counter = 0
        for batch in pbar:
            counter += 1

            phonePadded, tokenPadded, melPadded, phoneSeqLens, tokenSeqLens = batch
            phonePadded = phonePadded.cuda().long()
            tokenPadded = tokenPadded.cuda().long()
            melPadded = melPadded.cuda().float()
            phoneSeqLens = phoneSeqLens.cuda().long()
            tokenSeqLens = tokenSeqLens.cuda().long()

            # phonePadded = phonePadded.cpu().long()
            # tokenPadded = tokenPadded.cpu().long()
            # melPadded = melPadded.cpu().float()
            # phoneSeqLens = phoneSeqLens.cpu().long()
            # tokenSeqLens = tokenSeqLens.cpu().long()
            # model.cpu()

            simple_loss, pruned_loss = model(phonePadded,
                                             tokenPadded,
                                             phoneSeqLens,
                                             tokenSeqLens,
                                             melPadded,
                                             tokenPadded[:, 1:],
                                             warmup=1.0)

            # check if NaN
            simple_loss_is_finite = torch.isfinite(simple_loss)
            pruned_loss_is_finite = torch.isfinite(pruned_loss)
            is_finite = simple_loss_is_finite & pruned_loss_is_finite
            if not torch.all(is_finite):
                simple_loss = simple_loss[simple_loss_is_finite]
                pruned_loss = pruned_loss[pruned_loss_is_finite]

                # If either all simple_loss or pruned_loss is inf or nan,
                # we stop the training process by raising an exception
                if torch.all(~simple_loss_is_finite) or torch.all(~pruned_loss_is_finite):
                    raise ValueError(
                        "There are too many utterances in this batch "
                        "leading to inf or nan losses."
                    )

            simple_loss = simple_loss.sum()
            pruned_loss = pruned_loss.sum()
            losses = simple_loss * 0.5 + pruned_loss
            showLoss = losses.clone().detach()
            epochLoss += losses

            if counter == 10:
                break

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        lr_scheduler.step_batch(iteration)
        scaler.step(optimizer)
        scaler.update()

        batchNum += 1
        accu_iter += 1
        if rank == 0:
            pbar.set_description(
                "Train {} total_loss{:.4f}".format(accu_iter, showLoss.item()
                                                   ))

        # save
        if rank == 0 and (iteration % hparams.saved_checkpoint) == 0:
            print("save checkpint...")
            checkpoint_path = os.path.join(
                args.output_directory, "ckpt",
                "checkpoint_{}".format(iteration))
            save_checkpoint(
                model, optimizer, optimizer.param_groups[0]['lr'], iteration, checkpoint_path)
        if rank == 0 and (iteration % hparams.summary_interval == 0):
            sw.add_scalar("rnn_loss", showLoss, iteration)

        iteration += 1

        # Eval
        if rank == 0 and (iteration % hparams.evaluate_epoch == 0):
            model.eval()
            countNum = 0
            with torch.inference_mode():
                pbar2 = tqdm(validLoader)
                for i, batch in enumerate(pbar2):
                    phonePadded, tokenPadded, melPadded, phoneSeqLens, tokenSeqLens = batch
                    phonePadded = phonePadded.cuda().long()
                    tokenPadded = tokenPadded.cuda().long()
                    melPadded = melPadded.cuda().float()
                    phoneSeqLens = phoneSeqLens.cuda().long()
                    tokenSeqLens = tokenSeqLens.cuda().long()

                    if args.num_gpus > 1:
                        result = model.module.recognize(
                            inputs=phonePadded, input_lens=phoneSeqLens, reference_audio=melPadded)
                    else:
                        result = model.recognize(
                            inputs=phonePadded, input_lens=phoneSeqLens, reference_audio=melPadded)

                    countNum += 1
                    os.makedirs(os.path.join(args.output_directory,
                                str(accu_iter)), exist_ok=True)
                    np.save(os.path.join(args.output_directory, str(
                        accu_iter), str(i) + '.npy'), result.cpu().data.numpy)
                    if countNum == 5:
                        break

        epochLoss /= batchNum
        print("Epoch {}: Train {} total_loss: {:.4f}".format(
            epoch, accu_iter, epochLoss.item()))
        lr_scheduler.step()


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
