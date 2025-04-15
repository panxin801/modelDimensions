import torch
import copy
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from typing import (Optional, Tuple)


class Hubert(nn.Module):
    def __init__(self, num_label_embedding: int = 100,
                 mask: bool = True):
        super().__init__()

        self._mask = mask  # _mask is a private member

        self.feature_extractor = FeatureExtractor()
        self.feature_projection = FeatureProjection()
        self.positional_embedding = PositionalConvEmbedding()
        self.norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(nn.TransformerEncoderLayer(
            768, 12, 3072, activation="gelu", batch_first=True), 12)
        self.proj = nn.Linear(768, 256)

        # 用于在模型中表示被遮蔽的频谱嵌入（masked spectrum embedding）。在许多自监督学习任务中，特别是类似于 BERT 的掩码语言模型（Masked Language Model）的任务中，
        # 会随机遮蔽部分输入数据，并使用一个特定的嵌入向量来表示这些被遮蔽的部分。
        self.masked_spec_embed = nn.Parameter(
            torch.FloatTensor(768).uniform_())  # self.masked_spec_embed.size()=[768]
        self.label_embedding = nn.Embedding(num_label_embedding, 256)

    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
        x: [B,F,D] F means wav frames, D=768

        Return:
        x: [B,F,D] F means wav frames, D=768
        mask: [B,F] F means wav frames
        """
        mask = None
        if self.training and self._mask:
            mask = _compute_mask((x.size(0), x.size(1)), 0.8, 10, x.device, 2)
            # mask 部分用self.masked_spec_embed代替
            x[mask] = self.masked_spec_embed.to(x.dtype)
        return x, mask

    def encode(self, x: torch.Tensor, layer: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
        x: [B,1,T] T means wav samples.

        Return:
        x: [B,F,D] D=768
        mask: [B,F]
        """
        x = self.feature_extractor(x)  # x.size()=[B,D,F] D=512, F=frams
        x = self.feature_projection(
            x.transpose(1, 2))  # x.size()=[B,F,D] D=768
        x, mask = self.mask(x)  # x.size()=[B,F,D] D=768, mask.size()=[B,F]
        x = x + self.positional_embedding(x)  # x.size() = [B, F, D] D = 768
        x = self.dropout(self.norm(x))  # x.size() = [B, F, D] D = 768
        x = self.encoder(x, output_layer=layer)  # x.size() = [B, F, D] D = 768
        return x, mask

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x: [B,F,D] D=256, F=72 frams

        Return:
        logits: [B,F,D] D=100
        """
        logits = torch.cosine_similarity(x.unsqueeze(
            2), self.label_embedding.weight.unsqueeze(0).unsqueeze(0), dim=-1)
        return logits / 0.1  # logits.size()=[B,F,D] D=100

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
        x: [B,1,T] T means wav samples.

        Return:
        logits: [B,F,D] D=100
        mask: [B,F] F means wav frames
        """
        x, mask = self.encode(x)  # x.size() =[B,F,D], mask.size()=[B,F]
        x = self.proj(x)  # x.size() =[B,F,D]
        logits = self.logits(x)  # logits.size=[B,F,D] D=100
        return logits, mask


class HubertSoft(Hubert):
    """HuBERT-Soft content encoder from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`."""

    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def units(self, wav: torch.Tensor) -> torch.Tensor:
        """Extract soft speech units.
        Args:
            wav (Tensor): an audio waveform of shape (1, 1, T), where T is the number of samples.

        Returns:
            Tensor: soft speech units of shape (1, N, D), where N is the number of frames and D is the unit dimensions.
        """
        wav = F.pad(wav, ((100 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav)
        return self.proj(x)  # 相比Hubert没有计算logits


class HubertDiscrete(Hubert):
    def __init__(self, kmeans: KMeans):
        super().__init__(504)
        self.kmeans = kmeans

    @torch.inference_mode()
    def units(self, wav: torch.Tensor) -> torch.LongTensor:
        """Extract discrete speech units.
        Args:
            wav (Tensor): an audio waveform of shape (1, 1, T), where T is the number of samples.

        Returns:
            LongTensor: soft speech units of shape (N,), where N is the number of frames.
        """

        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav, layer=7)
        x = self.kmeans.predict(x.squeeze().cpu().numpy())
        return torch.tensor(x, dtype=torch.long, device=wav.device)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 512, 10, 5, bias=False)
        self.norm0 = nn.GroupNorm(512, 512)
        self.conv1 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv2 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv3 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv4 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv5 = nn.Conv1d(512, 512, 2, 2, bias=False)
        self.conv6 = nn.Conv1d(512, 512, 2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x: [B,1,T] T means wav samples.

        Return:
        x: [B,D,F] D=512, F=72 frams
        """
        x = F.gelu(self.norm0(self.conv0(x)))
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))
        x = F.gelu(self.conv6(x))
        return x


class FeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x: [B,F,D] F=72 frams, D=512

        Return:
        x: x.size()=[B,F,D] D=768
        """

        x = self.norm(x)  # x.size()=[B,F,D] D=512
        x = self.projection(x)  # x.size()=[B,F,D] D=768
        x = self.dropout(x)  # x.size()=[B,F,D] D=768
        return x


class PositionalConvEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(768,
                              768,
                              kernel_size=128,
                              padding=128 // 2,
                              groups=16,)
        self.conv = nn.utils.parametrizations.weight_norm(
            self.conv, name="weight", dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x: [B,F,D] F=72 frams, D=768

        Return:
        x: [B,F,D] F=73 frames, D=768
        """

        x = self.conv(x.transpose(1, 2))  # x.size()=[B,D,F] F=73 frames, D=768
        x = F.gelu(x[:, :, :-1])  # x.size()=[B,D,F] F=72 frames, D=768
        return x.transpose(1, 2)  # x.size()=[B,F,D] F=73 frames, D=768


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            *[copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )  # num_layers=12
        self.num_layers = num_layers

    def forward(self, src: torch.Tensor,
                mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None,
                output_layer: Optional[int] = None) -> torch.Tensor:
        """
        Args:
        src: [B,F,D] D=768, F=72 frams

        Return:
        output: [B,F,D] D=768
        """

        output = src
        for layer in self.layers[:output_layer]:  # self.layers[:None] 就是全部
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)
        return output  # [B,F,D] D=768


def _compute_mask(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    device: torch.device,
    min_masks: int = 0,
) -> torch.Tensor:
    """
        Args:
        shape: [B,F]=[32, 72],F=72 frams
        mask_prob: 0.8
        mask_lengthy: 10
        device: cuda:0
        min_masks: 2

        Return:
        mask: [B,F]=[32, 72], mask.dtype=torch.bool
        """
    batch_size, sequence_length = shape  # 32, 72

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length /
                           mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    mask = torch.zeros((batch_size, sequence_length),
                       device=device, dtype=torch.bool)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = torch.ones(
        (batch_size, sequence_length - (mask_length - 1)), device=device
    )

    # get random indices to mask
    mask_indices = torch.multinomial(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    mask_indices = (
        mask_indices.unsqueeze(dim=-1)
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    mask_idxs = mask_indices + offsets

    # scatter indices to mask
    mask = mask.scatter(1, mask_idxs, True)

    return mask  # mask.size()=[B,F]=[32, 72], mask.dtype=torch.bool
