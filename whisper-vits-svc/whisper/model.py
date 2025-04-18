import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from typing import (Iterable, Optional, Dict)
from dataclasses import dataclass

from .decoding import (detect_language as detect_language_function, decode as decode_function
                       )


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return super().forward(x.float()).type(x.dtype) sovits5.0
        return super().forward(x).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.type(x.dtype), None if self.bias is None else self.bias.type(x.dtype))


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        return super()._conv_forward(x, weight.type(x.dtype), None if bias is None else bias.type(x.dtype))


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment *
                               torch.arange(channels // 2))
    scaled_time = torch.arange(
        length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(self,
                x: torch.Tensor,
                xa: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[Dict] = None):
        q = self.query(x)

        if kv_cache is None or xa is None or not self.key in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        n_batch, n_ctx, n_state = q.size()
        scale = (n_state // self.n_head)**-0.5
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if not mask is None:
            qk += mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1, dtype=q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attention = MultiHeadAttention(
            n_state, n_head) if cross_attention else None
        self.cross_attention_ln = LayerNorm(
            n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(self,
                x: torch.Tensor,
                xa: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[Dict] = None
                ):
        x += self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attention:
            x += self.cross_attention(self.cross_attention_ln(x),
                                      xa, kv_cache=kv_cache)[0]
        x += self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3,
                            stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.Sequential(
            *[ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: torch.Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        len_x = x.size(1)
        len_e = self.positional_embedding.size(0)
        assert len_x <= len_e, "incorrect audio shape"
        pos_e = self.positional_embedding[:len_x, :]
        x = (x + pos_e).type(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.Sequential(
            *[ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        # 这一行代码的作用是将一个张量 mask 注册为模块（module）的缓冲区（buffer）。具体来说，这个方法有以下几个作用：
        # 缓冲区注册：缓冲区是模块的状态，但它们不会被更新（即不会在反向传播时计算梯度）。这与模型的参数（parameters）不同，参数会在训练过程中通过反向传播进行优化。
        # 持久性控制：通过设置 persistent=False，可以控制这个缓冲区是否在保存和加载模型时被持久化（即是否被保存到模型的状态字典中）。
        # persistent=False 表示这个缓冲区不会被持久化，这意味着当你保存模型的状态字典时，这个 mask 不会被包含在内。
        # 在你的代码中，mask 是一个用于自回归掩码的张量，通常在生成任务中用于防止模型在生成某个位置的输出时看到后续位置的信息。
        # 设置 persistent=False 的原因可能是因为这个 mask 是根据模型的上下文长度 n_ctx 动态生成的，并且在每次使用模型时，
        # 根据输入的上下文长度可能会重新生成，因此不需要将其保存为模型的状态的一部分。

    def forward(self, x: torch.Tensor, xa: torch.Tensor, kv_cache: Optional[Dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).size(1) if kv_cache else 0
        x = self.token_embedding(
            x) + self.positional_embedding[offset:offset + x.size(-1)]
        x = x.type(xa.dtype)
        for block in self.blocks:
            x = self.block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.type(x.dtype), 0, 1)).float()

        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat(
                    [cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    decode = decode_function
