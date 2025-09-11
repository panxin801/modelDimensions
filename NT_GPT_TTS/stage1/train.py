import argparse
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import json
import warnings

from utils import (Hparams, count_parameters)
from dataset import (TransducerDataset, TransducerCollate)
from stage1 import Stage1Net

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


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
                                     "train")
    validDataset = TransducerDataset(args.mel_scp,
                                     args.token_scp,
                                     args.valid_meta,
                                     "valid")
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
