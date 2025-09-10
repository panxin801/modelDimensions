import argparse
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import json
import warnings

from utils import Hparams
from dataset import (TransducerDataset, TransducerCollate)

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
        torch.cuda.manual_seed_all(hparams.seed)

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
