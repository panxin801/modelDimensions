# Copyright    2023                             (authors: Feiteng Li)
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
import math
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self,
                 dim_model: int,  # 1024
                 vocab_size: int,
                 dropout: float = 0.0):

        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model

        self.dropout = nn.Dropout(dropout)
        self.word_embeddings = nn.Embedding(vocab_size, dim_model)

    @property
    def weight(self):
        return self.word_embeddings.weight

    def embedding(self, index: int) -> torch.Tensor:
        return self.word_embeddings[index: index + 1]

    def forward(self, x: torch.Tensor):
        x = self.word_embeddings(x)
        x = self.dropout(x)

        return x


class SinePositionalEmbedding(nn.Module):
    def __init__(self,
                 dim_model: int,  # 1024
                 dropout: float = 0.0,  # 0.1
                 scale: bool = False,  # False
                 alpha: bool = False):  # True for AR, False for NAR

        super().__init__()

        self.dim_model = dim_model
        self.x_scale = math.sqrt(dim_model) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = nn.Dropout(dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 4000))

    def extend_pe(self, x):
        """Reset the positional encodings.
            Args: x:[1,4000]
        """

        if not self.pe is None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.dim_model)  # [4000,1024]
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(
                0, x.size(1), dtype=torch.float32
            ).unsqueeze(1)  # [x.size(1),1]=[4000,1]
        div_term = torch.exp(
            torch.arange(0, self.dim_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim_model)
        )  # [self.dim_model//2]=[512]
        # [x.size(1),self.dim_model//2]=[4000,512],处理的是偶数位置
        pe[:, 0::2] = torch.sin(position * div_term)
        # [x.size(1),self.dim_model//2]=[4000,512],处理的是奇数位置
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1,x.size(1),self.dim_model]=[1,4000,1024]
        self.pe = pe.to(device=x.device, dtype=x.dtype).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Args: x text or audio tokens [B, len, D]
        Return:
            output: tokens with positional encoding [B, len, D]
        """
        self.extend_pe(x)

        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * self.pe[:, :x.size(1)]
        output = self.dropout(output)
        return output
