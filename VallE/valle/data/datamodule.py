# Copyright      2023                          (authors: Feiteng Li)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import inspect  # 主要用于获取关于活动对象（如模块、类、方法、函数、回溯、帧对象等）的详细信息。它可以帮助开发者在运行时检查对象的定义、源代码、参数等，常用于调试、文档生成和一些高级的开发工具中。在 datamodule.py 中导入 inspect 可能是为了在代码的某个部分使用这些功能，比如检查某个对象的源代码或获取函数的参数信息。
import logging
import torch
from torch.utils.data import DataLoader
from functools import lru_cache
from pathlib import Path
from typing import (Any, Dict, Optional)
from icefall.utils import str2bool
from lhotse import (CutSet, load_manifest)
from lhotse.dataset import (CutConcatenate,
                            DynamicBucketingSampler,
                            PrecomputedFeatures,
                            SimpleCutSampler,
                            SpecAugment,)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from lhotse.utils import fix_random_seed


class TtsDataModule:
    """
    DataModule for VALL-E TTS experiments.
    It assumes there is always one train and valid dataloader.

    It contains all the common data pipeline modules used in TTS
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation[not used & tested yet],
    - augmentation[not used & tested yet],
    - on-the-fly feature extraction[not used & tested yet]

    This class should be derived for specific corpora used in TTS tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="TTS data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )  # Add a named group to parser, use parer._action_groups to get all groups
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/tokenized"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=40.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=10,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="When enabled, utterances (cuts) will be concatenated "
            "to minimize the amount of padding.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=0.1,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="When enabled, use on-the-fly cut mixing and feature "
            "extraction. Will drop existing precomputed feature manifests "
            "if available.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--buffer-size",
            type=int,
            default=40000,
            help="How many cuts (or cut pairs, triplets) we hold at any time across all of the buckets."
            "Increasing ``max_duration`` (batch_size) or ``num_buckets`` might require increasing this number."
            "It will result in larger memory usage.",
        )
        group.add_argument(
            "--shuffle-buffer-size",
            type=int,
            default=100000,
            help="How many cuts (or cut pairs, triplets) are being held in memory"
            "a buffer used for streaming shuffling. Larger number means better randomness at the cost"
            "of higher memory usage.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=False,
            help="Whether to drop last batch. Used by sampler.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['supervisions']['cut'] with the cuts that "
            "were used to construct it.",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )

        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=False,
            help="When enabled, use SpecAugment for training dataset.",
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )

        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures or PromptedPrecomputedFeatures",
        )

        group.add_argument(
            "--dataset",
            type=str,
            default="libritts",
            help="--input-strategy PromptedPrecomputedFeatures needs dataset name to prepare prompts.",
        )

        parser.add_argument(
            "--text-tokens",
            type=str,
            default="data/tokenized/unique_text_tokens.k2symbols",
            help="Path to the unique text tokens file",
        )

        parser.add_argument(
            "--sampling-rate",
            type=int,
            default=24000,
            help="""Audio sampling rate.""",
        )
