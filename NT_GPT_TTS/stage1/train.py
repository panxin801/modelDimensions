import os
import sys
import argparse
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import numpy as np
from warnings import simplefilter
from tqdm import tqdm


from utils import Hparams

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)


def train(rank, args, hparams):
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--train-meta", type=str, default="total.txt")
    parser.add_argument("--valid-meta", type=str, default="total.txt")
    parser.add_argument("--mel-scp", type=str, default="mel16k.scp")
    parser.add_argument("--token-scp", type=str, default="token.scp")
    parser.add_argument("--load-iteration", type=bool, default=True)
    parser.add_argument("--output-directory", type=str, default="stagemodel")
    parser.add_argument("--config", type=str, default="libritts_config.yaml")

    args = parser.parse_args()

    with open(args.config, "rt", encoding="utf8") as fr:
        data = fr.read()
    config = json.loads(data)
    hparams = Hparams(**config)

    torch.backends.cudnn.cudnn_enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    torch.backends.cudnn.deterministic = hparams.cudnn_deterministic

    if torch.cuda.is_available():
        torch.cuda.manual_seed(hparams.seed)
        torch.cuda.manual_seed_all(hparams.seed)
        torch.random.manual_seed(hparams.seed)
        np.random.seed(hparams.seed)

        args.num_gpus = torch.cuda.device_count()
        print("Number of GPUs:", args.num_gpus)
        args.batch_size = int(hparams.batch_size / args.num_gpus)
        print("Batch size per GPU:", args.batch_size)

    os.makedirs(os.path.join(args.output_directory, "ckpt"), exist_ok=True)
    if args.num_gpus > 1:
        mp.spawn(train, nprocs=args.num_gpus, args=(args, hparams))
    else:
        train(0, args, hparams)
