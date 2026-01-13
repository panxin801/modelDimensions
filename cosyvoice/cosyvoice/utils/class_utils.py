# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
#            2024 Alibaba Inc (authors: Xiang Lyu)
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

import torch
import torch.nn as nn

from cosyvoice.transformer.activation import Swish
from cosyvoice.transformer.subsampling import (
    LinearNoSubsampling,
    EmbedingNoSubsampling,
    Conv1dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    LegacyLinearNoSubsampling
)
from cosyvoice.transformer.embedding import (PositionalEncoding,
                                             RelPositionalEncoding,
                                             WhisperPositionalEncoding,
                                             LearnablePositionalEncoding,
                                             NoPositionalEncoding,
                                             EspnetRelPositionalEncoding)
from cosyvoice.transformer.attention import (MultiHeadedAttention,
                                             RelPositionMultiHeadedAttention)

COSYVOICE_ACTIVATION_CLASSES = {
    "hardtanh": nn.Hardtanh,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "swish": getattr(nn, "SiLU", Swish),
    "gelu": nn.GELU,
}

COSYVOICE_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "linear_legacy": LegacyLinearNoSubsampling,
    "embed": EmbedingNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
    'paraformer_dummy': nn.Identity
}


COSYVOICE_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "rel_pos_espnet": EspnetRelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
}
COSYVOICE_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
}


def get_model_type(configs):
    ...
