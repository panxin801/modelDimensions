import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import random
import contextlib
import warnings
warnings.filterwarnings("ignore")
from torch.cuda.amp import autocast


from module import (common, modules, attentions)
from text import symbols as symbols_v1
from text import symbols2 as symbols_v2


class SynthesizerTrnV3(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        semantic_frame_rate=None,
        freeze_quantizer=None,
        version="v3",
        **kwargs,
    ):
        super().__init__()

        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.version = version

        self.model_dim = 512
        self.use_sdp = use_sdp
        self.enc_p = TextEncoder(inter_channels, hidden_channels,
                                 filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        # self.ref_enc = modules.MelStyleEncoder(spec_channels, style_vector_dim=gin_channels)###Rollback
        self.ref_enc = modules.MelStyleEncoder(
            704, style_vector_dim=gin_channels)  # Rollback
