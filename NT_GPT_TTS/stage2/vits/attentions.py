import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import commons
from modules import LayerNorm


class Encoder(nn.Module):
    def __init__(self,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size=1,
                 p_dropout=0.,
                 window_size=4,
                 **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(self.p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for i in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(self.hidden_channels,
                                                       self.hidden_channels,
                                                       self.n_heads,
                                                       p_dropout=self.p_dropout,
                                                       window_size=self.window_size))
            # 为什么是layernorm？因为输入文本长度不一样
            self.norm_layers_1.append(LayerNorm(self.hidden_channels))
            self.ffn_layers.append(FFN(self.hidden_channels,
                                       self.hidden_channels,
                                       self.filter_channels,
                                       self.kernel_size,
                                       p_dropout=self.p_dropout))
            self.norm_layers_2.append(LayerNorm(self.hidden_channels))

    def forward(self, x, x_mask):
        """
        Args:
        x: [B,H,T]=[1,192,199]
        x_mask: [B,1,T]=[1,1,199]

        Return:
        x: [B,H,T]=[1,192,199]
        """
        attn_mask = x_mask.unsqueeze(
            2) * x_mask.unsqueeze(-1)  # attn_mask.size()=[1,1,H,H]=[1,1,199,199]
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 channels,
                 out_channels,
                 n_heads,
                 p_dropout=0.,
                 window_size=None,
                 heads_share=True,
                 block_length=None,
                 proximal_bias=False,
                 proximal_init=False,
                 ):
        super().__init__()
        assert channels % n_heads == 0

        self. channels = channels  # 192
        self.out_channels = out_channels  # 192
        self.n_heads = n_heads  # 2
        self.p_dropout = p_dropout  # 0.1
        self.window_size = window_size  # 4， self-attention相对位置编码的窗口大小
        self.heads_share = heads_share  # True
        self.block_length = block_length  # None
        self.proximal_bias = proximal_bias  # False
        self.proximal_init = proximal_init  # False
        self.attn = None

        self.k_channels = channels // n_heads  # 每个头处理的特征数量，这里是96
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(self.p_dropout)

        if not self.window_size is None:
            n_head_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(torch.randn(
                n_head_rel, self.window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(
                n_head_rel, self.window_size * 2 + 1, self.k_channels) * rel_stddev)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        # 用于控制是否进行近邻初始化（proximity initialization）。
        # 在self_attention中，近邻初始化是一种初始化技巧，旨在让模型在初始训练阶段更好地捕捉局部信息。
        # 具体来说，近邻初始化会将查询（query）和键（key）的权重初始化为相同，
        # 从而使得模型在初始阶段更倾向于关注局部位置的信息。
        # 这样可以有助于模型在开始训练时更好地对齐和处理相邻的音素或词，从而加速收敛。
        if self.proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)

        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels,
                           t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels,
                           t_s).transpose(2, 3)

        scores = torch.matmul(
            query / math.sqrt(self.k_channels), key.transpose(-2, -1))  # q/sqrt(dim)*k.T

        # self-attention相对位置编码
        if not self.window_size is None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_k, t_s)  # 构建相对位置编码矩阵
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings)
            score_logits = self._relative_position_to_absolute_position(
                rel_logits)
            scores = scores + score_logits
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype)
        if not mask is None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if not self.block_length is None:
                assert t_s == t_t, "Local attention is only available for self-attention."
                block_mask = torch.ones_like(
                    scores).triu(-self.block_length).tril(self.block_length)
                scores = scores.mask_fill(block_mask == 0, -1e4)

        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)

        if not self.window_size is None:
            relative_weights = self._absolute_position_to_relative_position(
                p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings)

        output = output.transpose(2, 3).contiguous().view(
            b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            paded_relative_embeddings = F.pad(relative_embeddings,
                                              commons.convert_pad_shape(
                                                  [[0, 0], [pad_length, pad_length], [0, 0]]))
        else:
            paded_relative_embeddings = relative_embeddings
        used_relative_embeddings = paded_relative_embeddings[:,
                                                             slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, commons.convert_pad_shape(
            [[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, commons.convert_pad_shape(
            [[0, 0], [0, 0], [0, length - 1]]))

        # Reshape and slice out the padded elements.
        x_final = x_flat.view(
            [batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(x, commons.convert_pad_shape(
            [[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, commons.convert_pad_shape(
            [[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):  # Feed forward network
    def __init__(self,
                 in_channels,
                 out_channels,
                 filter_channels,
                 kernel_size,
                 p_dropout=0.,
                 activation=None,
                 causal=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(self.p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        paddings = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(paddings))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        paddings = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(paddings))
        return x
