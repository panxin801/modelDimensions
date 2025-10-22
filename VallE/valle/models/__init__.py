import argparse
import torch.nn as nn
from icefall.utils import (str2bool, AttributeDict)


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model-name",
        type=str,
        default="VALL-E",
        help="VALL-E, VALL-F, Transformer.",
    )
    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=1024,
        help="Embedding dimension in the decoder model.",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=16,
        help="Number of attention heads in the Decoder layers.",
    )
    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=12,
        help="Number of Decoder layers.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Model scale factor which will be assigned different meanings in different models.",
    )
    parser.add_argument(
        "--norm-first",
        type=str2bool,
        default=True,
        help="Pre or Post Normalization.",
    )
    parser.add_argument(
        "--add-prenet",
        type=str2bool,
        default=False,
        help="Whether add PreNet after Inputs.",
    )

    # VALL-E & F
    parser.add_argument(
        "--prefix-mode",
        type=int,
        default=0,
        help="The mode for how to prefix VALL-E NAR Decoder, "
        "0: no prefix, 1: 0 to random, 2: random to random, 4: chunk of pre or post utterance.",
    )
    parser.add_argument(
        "--share-embedding",
        type=str2bool,
        default=True,
        help="Share the parameters of the output projection layer with the parameters of the acoustic embedding.",
    )
    parser.add_argument(
        "--prepend-bos",
        type=str2bool,
        default=False,
        help="Whether prepend <BOS> to the acoustic tokens -> AR Decoder inputs.",
    )
    parser.add_argument(
        "--num-quantizers",
        type=int,
        default=8,
        help="Number of Audio/Semantic quantization layers.",
    )

    # Transformer
    parser.add_argument(
        "--scaling-xformers",
        type=str2bool,
        default=False,
        help="Apply Reworked Conformer scaling on Transformers.",
    )
