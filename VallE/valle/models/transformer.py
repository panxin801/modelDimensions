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
import torch.nn as nn
import torch.nn.functional as F
from icefall.utils import make_pad_mask
from torchmetrics.classification import BinaryAccuracy

from .valle import Transpose
from ..modules.embedding import (SinePositionalEmbedding, TokenEmbedding)
from ..modules.scaling import (BalancedDoubleSwish, ScaledLinear)
from ..modules.transformer import (
    BalancedBasicNorm,
    IdentityNorm,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from .macros import (NUM_MEL_BINS, NUM_TEXT_TOKENS)
from .visualizer import visualize

IdentityNorm = IdentityNorm


class Transformer(nn.Module):
    """It implements seq2seq Transformer TTS for debug(No StopPredictor and SpeakerEmbeding)
    Neural Speech Synthesis with Transformer Network
    https://arxiv.org/abs/1809.08895
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 norm_first: bool = True,
                 add_prenet: bool = False,
                 scaling_xformers: bool = False,):
        """
        Args:
        d_model:
        The number of expected features in the input (required).
        nhead:
        The number of heads in the multiheadattention models (required).
        num_layers:
        The number of sub-decoder-layers in the decoder (required).
        """
        super().__init__()

        self.text_embedding = TokenEmbedding(d_model, NUM_TEXT_TOKENS)  # W_x
