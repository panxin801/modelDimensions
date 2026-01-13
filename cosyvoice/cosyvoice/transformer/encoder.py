# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2024 Alibaba Inc (Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder definition."""
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from typing import Tuple

from cosyvoice.utils.class_utils import (
    COSYVOICE_EMB_CLASSES,
    COSYVOICE_SUBSAMPLE_CLASSES,
    COSYVOICE_ATTENTION_CLASSES,
    COSYVOICE_ACTIVATION_CLASSES,)


class BaseEncoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int = 256,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.0,
                 input_layer: str = "conv2d",
                 pos_enc_layer_type: str = "abs_pos",
                 normalize_before: bool = True,
                 static_chunk_size: int = 0,
                 use_dynamic_chunk: bool = False,
                 global_cmvn: torch.nn.Module = None,
                 use_dynamic_left_chunk: bool = False,
                 gradient_checkpointing: bool = False, ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
            key_bias: whether use bias in attention.linear_k, False for whisper models.
            gradient_checkpointing: rerunning a forward-pass segment for each
                checkpointed segment during backward.
        """
        super().__init__()

        self._output_size = output_size
        self.global_cmvn = global_cmvn
        self.normalize_before = normalize_before
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing

        self.embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size, positional_dropout_rate))
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)


class ConformerEncoder:
    ...
