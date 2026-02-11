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
from cosyvoice.utils.mask import (make_pad_mask, add_optional_chunk_mask)


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

    def forward(self,
                xs: torch.Tensor,
                xs_lens: torch.Tensor,
                decoding_chunk_size: int = 0,
                num_decoding_left_chunks: int = -1,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D), for inference is <prompt_text|target_text>, text embedding
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """

        """ For inference_sft the args are following:
            xs: <prompt_text|target_text>, text embedding, [1,T_text, 512]
            xs_lens: input length (B)=[T_text]
            self.use_dynamic_chunk=False
            self.use_dynamic_left_chunk=False
            decoding_chunk_size: 1
            self.static_chunk_size=1
            num_decoding_left_chunks: -1
        Return:
            xs: [1,T_text,1024]
            masks: [1,1,T_text]
        """
        T = xs.size(1)  # T_text
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(
            1)  # (B,1,T_text), True means useful
        if not self.global_cmvn is None:
            xs = self.global_cmvn(xs)
        # [1,T_text,1024], [1,2*T_text+2*offset-1,1024], [1, 1, T_text]
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B,1, T/subsample_rate)=(B,1,T_text), like len mask
        chunk_masks = add_optional_chunk_mask(xs,
                                              masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)  # [1, T_text, T_text], like attn mask
        if self.gradient_checkpointing and self.training:
            xs = self.forward_layers_checkpointed(
                xs,
                chunk_masks,
                pos_emb,
                mask_pad)
        else:
            xs = self.forward_layers(xs,  # [1,T_text,1024]
                                     chunk_masks,  # [1,T_text,T_text]
                                     pos_emb,  # [1, 2*T_text+2*offset-1, 1024]
                                     mask_pad)  # [1,1,T_text]
            # the upper line, xs has same shape as input xs
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_layers(self,
                       xs: torch.Tensor,
                       chunk_masks: torch.Tensor,  # like attn mask
                       pos_emb: torch.Tensor,
                       mask_pad: torch.Tensor,) -> torch.Tensor:  # like len mask
        """ Forward encoder layers.
        Args:
            xs: [B, T_text, 1024]
            chunk_masks: [1, T_text, T_text], 当前chunk_mask
            pos_emb: [B, 2*T_text+2*offset-1, 1024]
            mask_pad: [1,1,T_text], 全局padding mask
        Return:
            xs: [B, T_text, 1024]
        """
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs,
                                          chunk_masks,
                                          pos_emb,
                                          mask_pad)
        return xs

    @torch.jit.export
    def forward_chunk(self,
                      xs: torch.Tensor,
                      offset: int,
                      required_cache_size: int,
                      att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
                      cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
                      att_mask: torch.Tensor = torch.ones(
                          (0, 0, 0), dtype=torch.bool),  # 下三角矩阵（主对脚线以上为False）
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk, call from TransformerLM.forward_chunk()
        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`, known as [B,T_text, 1024]
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
        """

        assert xs.size(0) == 1
        # tmp_masks is just for interface compatibility
        tmp_masks = torch.ones(1, xs.size(
            1), device=xs.device, dtype=torch.bool)  # [1, T_text=107] for first chunk,  [1, T_text=1] for the rest
        # [1, 1, T_text] for the first chunk,  [1, 1, 1] for the rest
        tmp_masks = tmp_masks.unsqueeze(1)
        if not self.global_cmvn is None:
            xs = self.global_cmvn(xs)
        # NOTE(xcsong): Before embed, shape(xs) is (b=1, time, mel-dim)
        # [1, T_text, 1024], [1, 2*T_text+2*offset-1, 1024] for the first chunk
        # [1, 1, 1024], [1, 2*T_text+2*offset-1, 1024] for the rest
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
        # NOTE(xcsong): After  embed, shape(xs) is (b=1, chunk_size, hidden-dim)
        # 0, 0 for first chunk
        # 14, offset for second chunk and later
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        # T_text for first chunk, 1 for the rest
        chunk_size = xs.size(1)
        # T_text for first chunk, offset+1 for the rest
        attention_key_size = cache_t1 + chunk_size
        # [1, 2*T_text+2*offset-1, 1024]
        pos_emb = self.embed.position_encoding(offset=offset - cache_t1,
                                               size=attention_key_size)
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)
        r_att_cache = []
        r_cnn_cache = []
        # TransformerLM是14层transformer,self_attn+ffn+dropout 这么简单理解吧
        for i, layer in enumerate(self.encoders):
            # NOTE(xcsong): Before layer.forward
            #   shape(att_cache[i:i + 1]) is (1, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
            # First chunk
            # xs: [1, T_text=107, 1024],
            # att_mask: [1, T_text, T_text]
            # pos_emb: [1, 2*T_text+2*offset-1, 1024] 213
            # att_cache: [0,0,0,0]
            # cnn_cache: [0,0,0,0]
            # Return
            # xs: [1, T_text, 1024]
            # new_att_cache: [14, 16, T_text+offset, 128]
            # new_cnn_cache: [14,0,0,0]

            # Second chunk and later
            # xs: [1, 1, 1024]
            # att_mask: [1, 1, 1]
            # pos_emb: [1, 2*T_text+2*offset-1, 1024] 215
            # att_cache: [1,16,offset,128] 107
            # cnn_cache: [1,0,0,0]
            # Return
            # xs: [1, 1, 1024]
            # new_att_cache: [14, 16, T_text+offset, 128] 1+107
            # new_cnn_cache: [14, 0, 0, 0]
            xs, _, new_attn_cache, new_cnn_cache = layer(xs,
                                                         att_mask,
                                                         pos_emb,
                                                         att_cache=att_cache[i:i +
                                                                             1] if elayers > 0 else att_cache,
                                                         cnn_cache=cnn_cache[i:i + 1] if cnn_cache.size(0) > 0 else cnn_cache)
            # NOTE(xcsong): After layer.forward
            #   shape(new_att_cache) is (1, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (b=1, hidden-dim, cache_t2)
            r_att_cache.append(new_attn_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))
        if self.normalize_before:
            xs = self.after_norm(xs)

        # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        # [14,16,T_text+offset,128]
        r_att_cache = torch.cat(r_att_cache, dim=0)
        # NOTE(xcsong): shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
        # [14,0,0,0]
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)

        return (xs, r_att_cache, r_cnn_cache)

    @torch.jit.unused
    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward input chunk by chunk with chunk_size like a streaming
            fashion
        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not preferred.
        Args:
            xs (torch.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]
            (y, att_cache, cnn_cache) = self.forward_chunk(
                chunk_xs,
                offset,
                required_cache_size,
                att_cache,
                cnn_cache)
            outputs.append(y)
            offset += y.size(1)
        ys = torch.cat(outputs, dim=1)
        masks = torch.cat((1, 1, ys.size(1)),
                          device=xs.device,
                          dtype=torch.bool)
        return ys, masks


class TransformerEncoder(BaseEncoder):
    """Transformer encoder module."""

    def __init__(self,
                 input_size: int,  # 1024
                 output_size: int = 256,  # 1024
                 attention_heads: int = 4,  # 16
                 linear_units: int = 2048,  # 4096
                 num_blocks: int = 6,  # 14
                 dropout_rate: float = 0.1,  # 0.1
                 positional_dropout_rate: float = 0.1,  # 0.1
                 attention_dropout_rate: float = 0.0,  # 0.0
                 input_layer: str = "conv2d",  # "linear_legacy"
                 pos_enc_layer_type: str = "abs_pos",  # rel_pos_espnet
                 normalize_before: bool = True,  # True
                 static_chunk_size: int = 0,  # 1
                 use_dynamic_chunk: bool = False,  # False
                 global_cmvn: torch.nn.Module = None,  # None
                 use_dynamic_left_chunk: bool = False,  # False
                 key_bias: bool = True,  # True
                 selfattention_layer_type: str = "selfattn",  # rel_selfattn
                 activation_type: str = "relu",  # "relu"
                 gradient_checkpointing: bool = False,  # False
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
    """Conformer encoder module.
    Used in LLM text_encoder, MaskedDiffWithXvec encoder
    """

    def __init__(self,
                 input_size: int,  # 512
                 output_size: int = 256,  # 1024
                 attention_heads: int = 4,  # 16
                 linear_units: int = 2048,  # 4096
                 num_blocks: int = 6,  # 6
                 dropout_rate: float = 0.1,  # 0.1
                 positional_dropout_rate: float = 0.1,  # 0.1
                 attention_dropout_rate: float = 0.0,  # 0.0
                 input_layer: str = "conv2d",  # "linear"
                 pos_enc_layer_type: str = "rel_pos",  # "rel_pos_espnet"
                 normalize_before: bool = True,  # True
                 static_chunk_size: int = 0,  # 1
                 global_cmvn: torch.nn.Module = None,  # None
                 use_dynamic_chunk: bool = False,  # False
                 use_dynamic_left_chunk: bool = False,  # False
                 positionwise_conv_kernel_size: int = 1,  # 1
                 macaron_style: bool = True,  # False
                 selfattention_layer_type: str = "rel_selfattn",  # "rel_selfattn"
                 activation_type: str = "swish",  # "swish"
                 use_cnn_module: bool = True,  # False
                 cnn_module_kernel: int = 15,  # 15
                 causal: bool = False,  # False
                 cnn_module_norm: str = "batch_norm",  # "batch_norm"
                 key_bias: bool = True,  # True
                 gradient_checkpointing: bool = False,  # False
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

        super().__init__(input_size,  # 512
                         output_size,  # 1024
                         attention_heads,  # 16
                         linear_units,  # 4096
                         num_blocks,  # 6
                         dropout_rate,  # 0.1
                         positional_dropout_rate,  # 0.1
                         attention_dropout_rate,  # 0.0
                         input_layer,  # "linear"
                         pos_enc_layer_type,  # "rel_pos_espnet"
                         normalize_before,  # True
                         static_chunk_size,  # 1
                         use_dynamic_chunk,  # False
                         global_cmvn,  # None
                         use_dynamic_left_chunk,  # False
                         gradient_checkpointing)  # False

        # "swish"-> "Silu"
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()

        # self-attention module args definition
        encoder_selfattn_layer = (attention_heads,  # 16
                                  output_size,  # 1024
                                  attention_dropout_rate,  # 0.0
                                  key_bias)  # True
        # feed-forward module args definition
        positionwise_layer_args = (output_size,  # 1024
                                   linear_units,  # 4096
                                   dropout_rate,  # 0.1
                                   activation)  # Silu()
        # convolution module definition
        convolution_layer_args = (output_size,  # 1024
                                  cnn_module_kernel,  # 15
                                  activation,  # Silu()
                                  cnn_module_norm,  # "batch_norm"
                                  causal)  # False

        self.encoders = nn.ModuleList([
            ConformerEncoderLayer(output_size,  # 1024
                                  COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](
                                      *encoder_selfattn_layer),
                                  PositionwiseFeedForward(
                                      *positionwise_layer_args),
                                  PositionwiseFeedForward(
                                      *positionwise_layer_args) if macaron_style else None,  # False
                                  ConvolutionModule(
                                      *convolution_layer_args) if use_cnn_module else None,  # False
                                  dropout_rate,  # 0.1
                                  normalize_before,  # True
                                  ) for _ in range(num_blocks)  # 6
        ])
