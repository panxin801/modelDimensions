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
from cosyvoice.transformer.encoder_layer import (
    ConformerEncoderLayer, TransformerEncoderLayer)
from cosyvoice.transformer.positionwise_feed_forward import PositionwiseFeedForward
from cosyvoice.transformer.convolution import ConvolutionModule


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

    def output_size(self,) -> int:
        return self._output_size


class TransformerEncoder(BaseEncoder):
    """Transformer encoder module."""

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
                 key_bias: bool = True,
                 selfattention_layer_type: str = "selfattn",
                 activation_type: str = "relu",
                 gradient_checkpointing: bool = False,
                 ):
        """ Construct TransformerEncoder
        See Encoder for the meaning of each parameter.
        """
        super().__init__(input_size,
                         output_size,
                         attention_heads,
                         linear_units,
                         num_blocks,
                         dropout_rate,
                         positional_dropout_rate,
                         attention_dropout_rate,
                         input_layer,
                         pos_enc_layer_type,
                         normalize_before,
                         static_chunk_size,
                         use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk,
                         gradient_checkpointing)
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(output_size,
                                    COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](attention_heads,
                                                                                          output_size,
                                                                                          attention_dropout_rate,
                                                                                          key_bias),
                                    PositionwiseFeedForward(output_size,
                                                            linear_units,
                                                            dropout_rate,
                                                            activation),
                                    dropout_rate, normalize_before) for _ in range(num_blocks)
        ])


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""

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
                 pos_enc_layer_type: str = "rel_pos",
                 normalize_before: bool = True,
                 static_chunk_size: int = 0,
                 use_dynamic_chunk: bool = False,
                 global_cmvn: torch.nn.Module = None,
                 use_dynamic_left_chunk: bool = False,
                 positionwise_conv_kernel_size: int = 1,
                 macaron_style: bool = True,
                 selfattention_layer_type: str = "rel_selfattn",
                 activation_type: str = "swish",
                 use_cnn_module: bool = True,
                 cnn_module_kernel: int = 15,
                 causal: bool = False,
                 cnn_module_norm: str = "batch_norm",
                 key_bias: bool = True,
                 gradient_checkpointing: bool = False,
                 ):
        """Construct ConformerEncoder
        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
            key_bias: whether use bias in attention.linear_k, False for whisper models.
        """

        super().__init__(input_size,
                         output_size,
                         attention_heads,
                         linear_units,
                         num_blocks,
                         dropout_rate,
                         positional_dropout_rate,
                         attention_dropout_rate,
                         input_layer,
                         pos_enc_layer_type,
                         normalize_before,
                         static_chunk_size,
                         use_dynamic_chunk,
                         global_cmvn,
                         use_dynamic_left_chunk,
                         gradient_checkpointing)
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()

        # self-attention module args definition
        encoder_selfattn_layer = (attention_heads,
                                  output_size,
                                  attention_dropout_rate,
                                  key_bias)
        # feed-forward module args definition
        positionwise_layer_args = (output_size,
                                   linear_units,
                                   dropout_rate,
                                   activation)
        # convolution module definition
        convolution_layer_args = (output_size,
                                  cnn_module_kernel,
                                  activation,
                                  cnn_module_norm,
                                  causal)

        self.encoders = nn.ModuleList([
            ConformerEncoderLayer(output_size,
                                  COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](
                                      *encoder_selfattn_layer),
                                  PositionwiseFeedForward(
                                      *positionwise_layer_args),
                                  PositionwiseFeedForward(
                                      *positionwise_layer_args) if macaron_style else None,
                                  ConvolutionModule(
                                      *convolution_layer_args) if use_cnn_module else None,
                                  dropout_rate,
                                  normalize_before,
                                  ) for _ in range(num_blocks)
        ])
