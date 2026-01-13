# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
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
"""Subsampling layer definition."""

import torch
import torch.nn as nn
from typing import (Tuple, Union)


class BaseSubsampling(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        # call sub class pos_enc.position_encoding()
        return self.pos_enc.position_encoding(offset, size)


class EmbedingNoSubsampling(BaseSubsampling):
    """Embedding input without subsampling
    """

    def __init__(self, idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Module):
        super().__init__()

        self.embed = nn.Embedding(idim, odim)
        self.pos_enc = pos_enc_class

    def forward(self, x: torch.Tensor,
                x_mask: torch.Tensor,
                offset: Union[int, torch.Tensor, ] = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .
        """
        x = self.embed(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an linear object."""
        super().__init__()

        self.out = nn.Sequential(
            nn.Linear(idim, odim),
            nn.LayerNorm(odim, eps=1e-5),
            nn.Dropout(dropout_rate),
        )

        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(self, x: torch.Tensor,
                x_mask: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .
        """
        # linear-> layernorm-> dropout-> pos_enc
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class Conv1dSubsampling2(BaseSubsampling):
    """Convolutional 1D subsampling (to 1/2 length).
       It is designed for Whisper, ref:
       https://github.com/openai/whisper/blob/main/whisper/model.py
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an Conv1dSubsampling2 object."""
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(idim, odim, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(odim, odim, 3, 2, 1),
            nn.GELU(),
        )
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 2
        # 4 = (3 - 1) * 1 + (3 - 1) * 1
        self.right_context = 4

    def forward(self, x: torch.Tensor,
                x_mask: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.
            torch.Tensor: positional encoding
        """

        time = x.size(1)
        x = x.transpose(1, 2)  # (b,f,t)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (b,t,f)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, (time + 1) % 2::2]


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(odim * ((idim - 1) // 2 - 1) // 2, odim)
        )
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(self, x: torch.Tensor,
                x_mask: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)  # (b,c=1,t,f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, f * c))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an Conv2dSubsampling6 object."""
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 5, 3),
            nn.ReLU()
        )
        self.linear = nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 6
        # 10=(3-1)*1+(5-1)*2
        self.right_context = 10

    def forward(self, x: torch.Tensor,
                x_mask: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.
            torch.Tensor: positional encoding
        """

        x = x.unsqueeze(1)  # (b,c=1,t,f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, f * c))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 4::3]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(),
        )
        self.pos_enc = pos_enc_class
        self.linear = nn.Linear(
            odim * (((idim - 1) // 2 - 1) // 2 - 1) // 2, odim)
        self.subsampling_rate = 8
        # 14=(3-1)*1+(3-1)*2+(3-1)*4
        self.right_context = 14

    def forward(self, x: torch.Tensor,
                x_mask: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
            torch.Tensor: positional encoding
        """

        x = x.unsqueeze(1)  # (b,c=1,t,f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, f * c))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2][:, :, 2::2]


class LegacyLinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an linear object."""
        super().__init__()

        self.out = nn.Sequential(
            nn.Linear(idim, odim),
            nn.LayerNorm(odim, eps=1e-5),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

        self.pos_enc = pos_enc_class
        self.subsampling_rate = 1
        self.right_context = 0

    def forward(self, x: torch.Tensor,
                x_mask: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .
        """

        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask
