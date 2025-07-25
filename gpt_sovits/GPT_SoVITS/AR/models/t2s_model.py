# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from AR.models.utils import (
    dpo_loss,
    get_batch_logps,
    make_pad_mask,
    make_pad_mask_left,
    make_reject_y,
    sample,
    topk_sampling,
)
from AR.modules.embedding import (SinePositionalEmbedding, TokenEmbedding)
from AR.modules.transformer import (
    LayerNorm, TransformerEncoder, TransformerEncoderLayer)

default_config = {
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_head": 8,
    "num_layers": 12,
    "num_codebook": 8,
    "p_dropout": 0.0,
    "vocab_size": 1024 + 1,
    "phoneme_vocab_size": 512,
    "EOS": 1024,
}

# @torch.jit.script ## 使用的话首次推理会非常慢，而且推理速度不稳定
# Efficient implementation equivalent to the following:


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, H, L, S = query.size(0), query.size(1), query.size(-2), key.size(-2)
    if scale is None:
        scale_factor = torch.tensor(1 / math.sqrt(query.size(-1)))
    else:
        scale_factor = scale
    attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight.masked_fill_(attn_mask, 0)
        else:
            attn_mask[attn_mask != float("-inf")] = 0
            attn_mask[attn_mask == float("-inf")] = 1
            attn_weight.masked_fill_(attn_mask, 0)

    return attn_weight @ value


@torch.jit.script
class T2SMLP:
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def forward(self, x):
        x = F.relu(F.linear(x, self.w1, self.b1))
        x = F.linear(x, self.w2, self.b2)
        return x


@torch.jit.script
class T2SBlock:
    def __init__(
        self,
        num_heads,
        hidden_dim: int,
        mlp: T2SMLP,
        qkv_w,
        qkv_b,
        out_w,
        out_b,
        norm_w1,
        norm_b1,
        norm_eps1,
        norm_w2,
        norm_b2,
        norm_eps2,
    ):
        self.num_heads = num_heads
        self.mlp = mlp
        self.hidden_dim: int = hidden_dim
        self.qkv_w = qkv_w
        self.qkv_b = qkv_b
        self.out_w = out_w
        self.out_b = out_b
        self.norm_w1 = norm_w1
        self.norm_b1 = norm_b1
        self.norm_eps1 = norm_eps1
        self.norm_w2 = norm_w2
        self.norm_b2 = norm_b2
        self.norm_eps2 = norm_eps2

        self.false = torch.tensor(False, dtype=torch.bool)

    @torch.jit.ignore
    def to_mask(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor],):
        if padding_mask is None:
            return x

        if padding_mask.dtype == torch.bool:
            return x.masked_fill(padding_mask, 0)
        else:
            return x * padding_mask

    def process_prompt(self,
                       # xy_pos, [1,src_len,512] 全部文本token+参考音频token
                       x: torch.Tensor,
                       # [1,self.num_head,src_len,src_len]
                       attn_mask: torch.Tensor,
                       padding_mask: Optional[torch.Tensor] = None,
                       torch_sdpa: bool = True):
        q, k, v = F.linear(self.to_mask(x, padding_mask),
                           self.qkv_w, self.qkv_b).chunk(3, dim=-1)  # q=k=v=[1,src_len,512]

        batch_size = q.shape[0]
        q_len = q.shape[1]  # 165
        kv_len = k.shape[1]  # 165

        q = self.to_mask(q, padding_mask)  # [1,src_len,512]
        k_cache = self.to_mask(k, padding_mask)  # [1,src_len,512]
        v_cache = self.to_mask(v, padding_mask)  # [1,src_len,512]

        q = q.view(batch_size, q_len, self.num_heads, -
                   1).transpose(1, 2)  # [1,self.num_head,src_len,512/self.num_head=32]
        k = k.view(batch_size, kv_len, self.num_heads, -
                   1).transpose(1, 2)  # [1,self.num_head,src_len,32]
        v = v.view(batch_size, kv_len, self.num_heads, -
                   1).transpose(1, 2)  # [1,self.num_head,src_len,32]

        if torch_sdpa:
            # [1,self.num_head,src_len,512/self.num_head=32]
            attn = F.scaled_dot_product_attention(q, k, v, ~attn_mask)
        else:
            attn = scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).reshape(
            batch_size, q_len, -1)  # [1,src_len,512]
        attn = F.linear(self.to_mask(attn, padding_mask),
                        self.out_w, self.out_b)  # [1,src_len,512]

        x = x + attn  # [1,src_len,512]
        x = F.layer_norm(x, [self.hidden_dim], self.norm_w1,
                         self.norm_b1, self.norm_eps1)
        x = x + self.mlp.forward(x)
        x = F.layer_norm(x, [self.hidden_dim], self.norm_w2,
                         self.norm_b2, self.norm_eps2)
        return x, k_cache, v_cache  # all are [1,src_len,512],

    def decode_next_token(self,
                          x: torch.Tensor,  # 最后i一帧音频token, [1,1,512]
                          k_cache: torch.Tensor,  # [1,src_len+n,512] 第n次预测就加n
                          v_cache: torch.Tensor,  # [1,src_len+n,512] 第n次预测就加n
                          attn_mask: torch.Tensor = None,
                          torch_sdpa: bool = True,):
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(
            3, dim=-1)  # q=k=v=[1,1,512]

        k_cache = torch.cat([k_cache, k], dim=1)  # [1,165+n,512] 拼接了新的k
        v_cache = torch.cat([v_cache, v], dim=1)  # [1,165+n,512] 拼接了新的v

        batch_size = q.shape[0]  # 1
        q_len = q.shape[1]  # 1
        kv_len = k_cache.shape[1]

        # [1,self.num_head,1,512/self.num_head=32]
        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len,
                         self.num_heads, -1).transpose(1, 2)  # [1,self.num_head,165+n,32]
        v = v_cache.view(batch_size, kv_len,
                         self.num_heads, -1).transpose(1, 2)  # [1,self.num_head,165+n,32]

        if torch_sdpa:
            attn = F.scaled_dot_product_attention(
                q, k, v, (~attn_mask) if attn_mask is not None else None)  # [1,self.num_head,1,32]
        else:
            attn = scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)  # [1,1,512]
        attn = F.linear(attn, self.out_w, self.out_b)  # [1,1,512]

        x = x + attn
        x = F.layer_norm(x, [self.hidden_dim], self.norm_w1,
                         self.norm_b1, self.norm_eps1)
        x = x + self.mlp.forward(x)
        x = F.layer_norm(x, [self.hidden_dim], self.norm_w2,
                         self.norm_b2, self.norm_eps2)  # [1,1,512]
        # [1,1,512], [1,165+n,512] ,[1,165+n,512] 第n次预测就加n
        return x, k_cache, v_cache


@torch.jit.script
class T2STransformer:
    def __init__(self, num_blocks: int, blocks: List[T2SBlock]):
        self.num_blocks: int = num_blocks
        self.blocks = blocks

    def process_prompt(
        self,
        x: torch.Tensor,  # 全部文本token+参考音频token, [1, T_phone+T_semantic, 512]
        # [1, self.num_head, T_phone+T_semantic, T_phone+T_semantic]
        attn_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        # torch_sdpa 是一个布尔值，指示是否使用 PyTorch 内置的 scaled_dot_product_attention 函数
        torch_sdpa: bool = True,
    ):
        """处理文本和参考音频提示
        """
        k_cache: List[torch.Tensor] = []
        v_cache: List[torch.Tensor] = []
        for i in range(self.num_blocks):
            # 循环处理每个编码器模块
            x, k_cache_, v_cache_ = self.blocks[i].process_prompt(
                x, attn_mask, padding_mask, torch_sdpa)  # all are[1,src_len,512],
            k_cache.append(k_cache_)
            v_cache.append(v_cache_)
        # [1,src_len,512]. k_cache and v_cache are List[1,src_len,512]
        return x, k_cache, v_cache

    def decode_next_token(
        self,
        x: torch.Tensor,  # [1,1,512], 最后一帧的音频token
        # List[1,src_len+n,512], len=24， 第n次decode next token 就加n
        k_cache: List[torch.Tensor],
        v_cache: List[torch.Tensor],
        attn_mask: torch.Tensor = None,
        torch_sdpa: bool = True,
    ):
        for i in range(self.num_blocks):
            # 处理当前的token也就是x，然后返回x同时更新kv_cache
            x, k_cache[i], v_cache[i] = self.blocks[i].decode_next_token(
                x, k_cache[i], v_cache[i], attn_mask, torch_sdpa
            )
        return x, k_cache, v_cache


class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, norm_first=False, top_k=3):
        super().__init__()

        self.model_dim = config["model"]["hidden_dim"]  # 512
        self.embedding_dim = config["model"]["embedding_dim"]  # 512
        self.num_head = config["model"]["head"]  # 16
        self.num_layers = config["model"]["n_layer"]  # 24
        self.norm_first = norm_first  # False
        self.vocab_size = config["model"]["vocab_size"]  # 1025
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]  # 732
        self.p_dropout = config["model"]["dropout"]  # 0
        self.EOS = config["model"]["EOS"]  # 1024
        assert self.EOS == self.vocab_size - 1
        # should be same as num of kmeans bin
        # assert self.EOS == 1024
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(self.embedding_dim,
                                                self.phoneme_vocab_size,
                                                self.p_dropout)
        self.ar_text_position = SinePositionalEmbedding(self.embedding_dim,
                                                        dropout=0.1,
                                                        scale=False,
                                                        alpha=True)
        self.ar_audio_embedding = TokenEmbedding(self.embedding_dim,
                                                 self.vocab_size,
                                                 self.p_dropout)
        self.ar_audio_position = SinePositionalEmbedding(self.embedding_dim,
                                                         dropout=0.1,
                                                         scale=False,
                                                         alpha=True)
        self.h = TransformerEncoder(TransformerEncoderLayer(d_model=self.model_dim,
                                                            nhead=self.num_head,
                                                            dim_feedforward=self.model_dim * 4,
                                                            dropout=0.1,
                                                            batch_first=True,
                                                            norm_first=norm_first),
                                    num_layers=self.num_layers,
                                    norm=LayerNorm(self.model_dim) if norm_first else None)
        self.ar_predict_layer = nn.Linear(
            self.model_dim, self.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")

        self.ar_accuracy_metric = MulticlassAccuracy(self.vocab_size,
                                                     top_k=top_k,
                                                     average="micro",
                                                     multidim_average="global",
                                                     ignore_index=self.EOS,
                                                     )
        blocks = []
        for i in range(self.num_layers):
            layer = self.h.layers[i]
            t2smlp = T2SMLP(layer.linear1.weight,
                            layer.linear1.bias,
                            layer.linear2.weight,
                            layer.linear2.bias,
                            )
            block = T2SBlock(self.num_head,
                             self.model_dim,
                             t2smlp,
                             layer.self_attn.in_proj_weight,
                             layer.self_attn.in_proj_bias,
                             layer.self_attn.out_proj.weight,
                             layer.self_attn.out_proj.bias,
                             layer.norm1.weight,
                             layer.norm1.bias,
                             layer.norm1.eps,
                             layer.norm2.weight,
                             layer.norm2.bias,
                             layer.norm2.eps,)
            blocks.append(block)
        self.t2s_transformer = T2STransformer(self.num_layers, blocks)

    def make_input_data(self, x, x_lens, y, y_lens, bert_feature):
        # Only used when DPO activated.
        """
        x: phoneme_ids: [B, T_phone]
        x_lens: [B]
        y: semantic_ids, from vq_model.extract_latent  [B, T_semantic], T_semantic=71
        y_lens: [B]
        bert_feature: bert feature from text # [B, 1024, T_phone]
        """

        x = self.ar_text_embedding(x)  # [B,T_phone, 512]
        # [B,T_phone, 512]
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)  # [B,T_phone, 512]
        x_mask = make_pad_mask(x_lens)  # [B,T_phone]
        y_mask = make_pad_mask(y_lens)  # [B,T_semantic]
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)  # [B,T_semantic]

        # Training
        # AR Decoder
        # y, targets=[B,T_semantic]
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()  # T_phone
        y_len = y_lens.max()  # T_semantic
        y_emb = self.ar_audio_embedding(y)  # [B,T_semantic,512]
        y_pos = self.ar_audio_position(y_emb)  # [B,T_semantic,512]

        xy_padding_mask = torch.cat(
            [x_mask, y_mask], dim=1)  # [B,T_phone+T_semantic]
        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = F.pad(torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
                            (0, y_len),
                            value=True,)  # [T_phone, T_phone+T_semantic]
        # x_attn_mask[:, x_len]=False
        y_attn_mask = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=x.device), diagonal=1),
                            (x_len, 0),
                            value=False,)  # [T_semantic, T_phone+T_semantic]
        # [T_phone+T_semantic, T_phone+T_semantic]
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (ar_xy_padding_mask.view(bsz, 1, 1, src_len)
                            .expand(-1, self.num_head, -1, -1)
                            .reshape(bsz * self.num_head, 1, src_len))  # [self.num_head, B, src_len]
        xy_attn_mask = xy_attn_mask.logical_or(
            _xy_padding_mask)  # [self.num_head, src_len, src_len]
        new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask  # [self.num_head, src_len, src_len]
        # x 和完整的 y 一次性输入模型
        xy_pos = torch.concat([x, y_pos], dim=1)  # [B, src_len, 512]

        return xy_pos, xy_attn_mask, targets

    def forward(self, x, x_lens, y, y_lens, bert_feature):
        """
        x: phoneme_ids: [B, T_phone]
        x_lens: [B]
        y: semantic_ids, from vq_model.extract_latent  [B, T_semantic], T_semantic=71
        y_lens: [B]
        bert_feature: bert feature from text # [B, 1024, T_phone]
        """

        reject_y, reject_y_lens = make_reject_y(
            y, y_lens)  # reject_y=[B,88], reject_y_lens=[B]

        xy_pos, xy_attn_mask, targets = self.make_input_data(
            x, x_lens, y, y_lens, bert_feature)  # xy_pos=[B,T_phone+T_semantic, 512], xy_attn_mask=[B,T_phone+T_semantic,T_phone+T_semantic], targets=[B,T_semantic]

        xy_dec, _ = self.h((xy_pos, None),
                           mask=xy_attn_mask)  # [B, T_phone+T_semantic, 512]
        x_len = x_lens.max()  # T_phone
        logits = self.ar_predict_layer(
            xy_dec[:, x_len:])  # [B,T_semantic, 1025]

        ###### DPO #############
        reject_xy_pos, reject_xy_attn_mask, reject_targets = self.make_input_data(
            x, x_lens, reject_y, reject_y_lens, bert_feature)  # [B,107,512], [B, 107, 107], [B,88]

        reject_xy_dec, _ = self.h(
            (reject_xy_pos, None),
            mask=reject_xy_attn_mask,
        )  # [B,107,512]
        x_len = x_lens.max()
        reject_logits = self.ar_predict_layer(
            reject_xy_dec[:, x_len:])  # [B,88,1025]

        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum

        loss_1 = F.cross_entropy(logits.permute(
            0, 2, 1), targets, reduction="sum")  # item
        acc = self.ar_accuracy_metric(
            logits.permute(0, 2, 1).detach(), targets).item()  # item

        A_logits, R_logits = get_batch_logps(
            logits, reject_logits, targets, reject_targets)  # [1],[1]
        loss_2, _, _ = dpo_loss(A_logits, R_logits, 0,
                                0, 0.2, reference_free=True)  # item
        loss = loss_1 + loss_2

        return loss, acc

    def forward_old(self, x, x_lens, y, y_lens, bert_feature):
        """
        x: phoneme_ids: [B, T_phone]
        x_lens: [B]
        y: semantic_ids, from vq_model.extract_latent  [B, T_semantic]
        y_lens: [B]
        bert_feature: bert feature from text # [B, 1024, T_phone]
        """
        x = self.ar_text_embedding(x)  # [B,T_phone,512]
        x = x + self.bert_proj(bert_feature.transpose(1, 2))  # [B,T_phone,512]
        x = self.ar_text_position(x)  # [B,T_phone,512]

        x_mask = make_pad_mask(x_lens)  # [B, T_phone]
        y_mask = make_pad_mask(y_lens)  # [B, T_semantic]
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)  # [B, T_semantic]

        # Training
        # AR Decoder
        # y=targets=[B, T_semantic]
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()
        y_len = y_lens.max()
        y_emb = self.ar_audio_embedding(y)  # [B,T_semantic,512]
        y_pos = self.ar_audio_position(y_emb)  # [B,T_semantic,512]

        xy_padding_mask = torch.concat(
            [x_mask, y_mask], dim=1)  # [B,T_phone+T_audio]
        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )  # [x_len ,x_len+y_len]
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )  # [y_len,x_len+y_len]
        # [x_len+y_len,x_len+y_len]
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            ar_xy_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.num_head, -1, -1)
            .reshape(bsz * self.num_head, 1, src_len)
        )  # [self.num_head,B,src_len]
        xy_attn_mask = xy_attn_mask.logical_or(
            _xy_padding_mask)  # [self.num_head,src_len,src_len]
        # [self.num_head,src_len,src_len]
        new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask  # [self.num_head,src_len,src_len]
        # x 和完整的 y 一次性输入模型
        xy_pos = torch.concat([x, y_pos], dim=1)  # [B,T_phone+T_audio,512]
        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )  # [B,T_phone+T_audio,512]
        logits = self.ar_predict_layer(xy_dec[:, x_len:]).permute(
            0, 2, 1)  # [B,1025, T_audio]
        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum
        loss = F.cross_entropy(logits, targets, reduction="sum")
        acc = self.ar_accuracy_metric(logits.detach(), targets).item()
        return loss, acc

    # 需要看下这个函数和 forward 的区别以及没有 semantic 的时候 prompts 输入什么
    def infer(
        self,
        x,
        x_lens,
        prompts,
        bert_feature,
        top_k: int = -100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
    ):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)

        # AR Decoder
        y = prompts
        prefix_len = y.shape[1]
        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False
        for _ in tqdm(range(1500)):
            y_emb = self.ar_audio_embedding(y)
            y_pos = self.ar_audio_position(y_emb)
            # x 和逐渐增长的 y 一起输入给模型
            xy_pos = torch.concat([x, y_pos], dim=1)
            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(x_attn_mask, (0, y_len), value=True)
            y_attn_mask = F.pad(torch.triu(torch.ones(
                y_len, y_len, dtype=torch.bool), diagonal=1), (x_len, 0), value=False)
            xy_attn_mask = torch.concat(
                [x_attn_mask_pad, y_attn_mask], dim=0).to(y.device)
            xy_dec, _ = self.h((xy_pos, None), mask=xy_attn_mask)
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = topk_sampling(
                logits, top_k=top_k, top_p=1.0, temperature=temperature)

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                # print(torch.argmax(logits, dim=-1)[0] == self.EOS, samples[0, 0] == self.EOS)
                stop = True
            if stop:
                if prompts.shape[1] == y.shape[1]:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break
            # 本次生成的 semantic_ids 和之前的 y 构成新的 y
            # print(samples.shape)#[1,1]#第一个1是bs
            # import os
            # os._exit(2333)
            y = torch.concat([y, samples], dim=1)
        return y

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(y, (0, 1), value=0) + eos_id * \
            F.pad(y_mask_int, (0, 1), value=1)
        # 错位
        return targets[:, :-1], targets[:, 1:]

    def infer_panel_batch_infer(
        self,
        x: List[torch.LongTensor],  # 全部文本token
        x_lens: torch.LongTensor,
        prompts: torch.LongTensor,  # 参考音频token
        bert_feature: List[torch.LongTensor],
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs,
    ):
        if prompts is None:
            print(
                "Warning: Prompt free is not supported batch_infer! switch to naive_infer")
            return self.infer_panel_naive_batched(
                x,
                x_lens,
                prompts,
                bert_feature,
                top_k=top_k,
                top_p=top_p,
                early_stop_num=early_stop_num,
                temperature=temperature,
                **kwargs,
            )

        max_len = kwargs.get("max_len", x_lens.max())
        x_list = []
        for x_item, bert_item in zip(x, bert_feature):
            # max_len = max(max_len, x_item.shape[0], bert_item.shape[1])
            x_item = self.ar_text_embedding(x_item.unsqueeze(0))
            x_item = x_item + \
                self.bert_proj(bert_item.transpose(0, 1).unsqueeze(0))
            x_item = self.ar_text_position(x_item).squeeze(0)
            # x_item = F.pad(x_item,(0,0,0,max_len-x_item.shape[0]),value=0) if x_item.shape[0]<max_len else x_item  ### padding right
            # padding left
            x_item = (F.pad(x_item, (0, 0, max_len -
                      x_item.shape[0], 0), value=0) if x_item.shape[0] < max_len else x_item)
            x_list.append(x_item)
        x: torch.Tensor = torch.stack(x_list, dim=0)

        # AR Decoder
        y = prompts

        x_len = x.shape[1]
        stop = False

        k_cache = None
        v_cache = None
        ###################  first step ##########################
        assert y is not None, "Error: Prompt free is not supported batch_infer!"
        ref_free = False

        y_emb = self.ar_audio_embedding(y)
        y_len = y_emb.shape[1]
        prefix_len = y.shape[1]
        y_lens = torch.LongTensor(
            [y_emb.shape[1]] * y_emb.shape[0]).to(x.device)
        y_pos = self.ar_audio_position(y_emb)
        xy_pos = torch.concat([x, y_pos], dim=1)

        ##### create mask #####
        bsz = x.shape[0]
        src_len = x_len + y_len
        y_paddind_mask = make_pad_mask_left(y_lens, y_len)
        x_paddind_mask = make_pad_mask_left(x_lens, max_len)

        # (bsz, x_len + y_len)
        padding_mask = torch.concat([x_paddind_mask, y_paddind_mask], dim=1)

        x_mask = F.pad(
            torch.zeros(x_len, x_len, dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )

        y_mask = F.pad(  # yy的右上1扩展到左边xy的0,(y,x+y)
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool,
                       device=x.device), diagonal=1),
            (x_len, 0),
            value=False,
        )

        causal_mask = torch.concat([x_mask, y_mask], dim=0).view(
            1, src_len, src_len).repeat(bsz, 1, 1).to(x.device)
        # padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2) ### [b, x+y, x+y]
        # 上面是错误的，会导致padding的token被"看见"

        # 正确的padding_mask应该是：
        # |   pad_len   |  x_len  |  y_len  |
        # [[PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],  前3行按理说也应该被mask掉，但是为了防止计算attention时不出现nan，还是保留了，不影响结果
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6]]

        padding_mask = padding_mask.view(bsz, 1, src_len).repeat(1, src_len, 1)

        attn_mask: torch.Tensor = causal_mask.logical_or(padding_mask)
        attn_mask = attn_mask.unsqueeze(
            1).expand(-1, self.num_head, -1, -1).bool()

        # 正确的attn_mask应该是这样的：
        # |   pad_len   |  x_len  |  y_len  |
        # [[PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],  前3行按理说也应该被mask掉，但是为了防止计算attention时不出现nan，还是保留了，不影响结果
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3,   4, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3,   4,   5, EOS],
        # [PAD, PAD, PAD, 1, 2, 3,   4,   5,   6]]

        ###### decode #####
        y_list = [None] * y.shape[0]
        batch_idx_map = list(range(y.shape[0]))
        idx_list = [None] * y.shape[0]
        for idx in tqdm(range(1500)):
            if idx == 0:
                xy_dec, k_cache, v_cache = self.t2s_transformer.process_prompt(
                    xy_pos, attn_mask, None)
            else:
                xy_dec, k_cache, v_cache = self.t2s_transformer.decode_next_token(
                    xy_pos, k_cache, v_cache, attn_mask)
            logits = self.ar_predict_layer(xy_dec[:, -1])

            if idx == 0:
                attn_mask = F.pad(
                    attn_mask[:, :, -1].unsqueeze(-2), (0, 1), value=False)
                logits = logits[:, :-1]
            else:
                attn_mask = F.pad(attn_mask, (0, 1), value=False)

            samples = sample(
                logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature
            )[0]

            y = torch.concat([y, samples], dim=1)

            # 移除batch中已经生成完毕的序列,进一步优化计算量
            tokens = torch.argmax(logits, dim=-1)
            reserved_idx_of_batch_for_y = None
            if (self.EOS in samples[:, 0]) or (self.EOS in tokens):  # 如果生成到EOS，则停止
                l1 = samples[:, 0] == self.EOS
                l2 = tokens == self.EOS
                l = l1.logical_or(l2)
                removed_idx_of_batch_for_y = torch.where(l == True)[0].tolist()
                reserved_idx_of_batch_for_y = torch.where(l == False)[0]
                # batch_indexs = torch.tensor(batch_idx_map, device=y.device)[removed_idx_of_batch_for_y]
                for i in removed_idx_of_batch_for_y:
                    batch_index = batch_idx_map[i]
                    idx_list[batch_index] = idx
                    y_list[batch_index] = y[i, :-1]

                batch_idx_map = [batch_idx_map[i]
                                 for i in reserved_idx_of_batch_for_y.tolist()]

            # 只保留batch中未生成完毕的序列
            if reserved_idx_of_batch_for_y is not None:
                # index = torch.LongTensor(batch_idx_map).to(y.device)
                y = torch.index_select(
                    y, dim=0, index=reserved_idx_of_batch_for_y)
                attn_mask = torch.index_select(
                    attn_mask, dim=0, index=reserved_idx_of_batch_for_y)
                if k_cache is not None:
                    for i in range(len(k_cache)):
                        k_cache[i] = torch.index_select(
                            k_cache[i], dim=0, index=reserved_idx_of_batch_for_y)
                        v_cache[i] = torch.index_select(
                            v_cache[i], dim=0, index=reserved_idx_of_batch_for_y)

            if (early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num) or idx == 1499:
                print("use early stop num:", early_stop_num)
                stop = True
                for i, batch_index in enumerate(batch_idx_map):
                    batch_index = batch_idx_map[i]
                    idx_list[batch_index] = idx
                    y_list[batch_index] = y[i, :-1]

            if None not in idx_list:
                stop = True

            if stop:
                if y.shape[1] == 0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break

            ####################### update next step ###################################
            y_emb = self.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[
                :, y_len + idx
            ].to(dtype=y_emb.dtype, device=y_emb.device)

        if None in idx_list:
            for i in range(x.shape[0]):
                if idx_list[i] is None:
                    idx_list[i] = 1500 - 1  # 如果没有生成到EOS，就用最大长度代替

        if ref_free:
            return y_list, [0] * x.shape[0]
        # print(idx_list)
        return y_list, idx_list

    def infer_panel_naive_batched(
        self,
        x: List[torch.LongTensor],  # 全部文本token
        x_lens: torch.LongTensor,
        prompts: torch.LongTensor,  # 参考音频token
        bert_feature: List[torch.LongTensor],
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs,
    ):
        y_list = []
        idx_list = []
        for i in range(len(x)):
            y, idx = self.infer_panel_naive(
                x[i].unsqueeze(0),
                x_lens[i],
                prompts[i].unsqueeze(0) if prompts is not None else None,
                bert_feature[i].unsqueeze(0),
                top_k,
                top_p,
                early_stop_num,
                temperature,
                repetition_penalty,
                **kwargs,
            )
            y_list.append(y[0])
            idx_list.append(idx)

        return y_list, idx_list

    def infer_panel_naive(
        self,
        x: torch.LongTensor,  # 全部文本token, [1,T_phoneme=41]
        x_lens: torch.LongTensor,  # [1]=41
        prompts: torch.LongTensor,  # 参考音频token，来自SSL, [1,T_semantic=124]
        # 全部文本bert feature [1,1024, T_phoneme=41]
        bert_feature: torch.LongTensor,
        top_k: int = -100,  # 20
        top_p: int = 100,  # 1
        early_stop_num: int = -1,  # 2700
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs,
    ):
        x = self.ar_text_embedding(x)  # [1,T_phoneme,512],全部文本token转到嵌入向量表示
        # 全部文本token+bert特征，融合bert特征的文本嵌入向量表示
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        # [1,T_phoneme,512]。这对于自注意力机制来说是非常重要的，因为它使得模型能够理解token之间的相对顺序
        x = self.ar_text_position(x)

        # AR Decoder
        # 参考音频token, from ssl，是通过cnhubert计算的用于引导音频语义序列的生成,[1,T_semantic=124]
        y = prompts

        x_len = x.shape[1]  # T_phoneme，长度用于后续生成注意力掩码（attn_mask）
        # [T_phoneme,T_phoneme]，x_attn_mask用于在自注意力机制中避免文本token之间的未来信息泄露
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False  # 控制流程是否应该提前终止。

        k_cache = None  # 分别用于存储自注意力机制中的key和value的缓存
        v_cache = None
        ###################  first step ##########################
        if y is not None:  # 有参考音频
            # [1,T_semantic,512]，参考音频token转换到嵌入向量
            y_emb = self.ar_audio_embedding(y)
            y_len = y_emb.shape[1]  # T_semantic
            prefix_len = y.shape[1]  # T_semantic
            # [1,T_semantic,512]，对参考音频token嵌入向量添加位置编码，用于attention 理解token顺序。
            y_pos = self.ar_audio_position(y_emb)
            # [1,T_phoneme+T_semantic,512]
            # 全部文本token嵌入向量，参考音频token嵌入向量拼接一起
            xy_pos = torch.concat([x, y_pos], dim=1)
            ref_free = False
        else:
            y_emb = None
            y_len = 0
            prefix_len = 0
            y_pos = None
            xy_pos = x
            y = torch.zeros(x.shape[0], 0, dtype=torch.int, device=x.device)
            ref_free = True

        bsz = x.shape[0]
        src_len = x_len + y_len
        x_attn_mask_pad = F.pad(
            x_attn_mask,  # 文本序列的注意力掩码，初始时为全0矩阵
            (0, y_len),  # xx的纯0扩展到xx纯0+xy纯1，(x,x+y)
            value=True,
        )  # [T_phoneme, src_len],即在文本序列的末尾添加 y_len 列 True，使得文本序列不能看到音频序列的位置。x_attn_mask_pad是文本序列的注意力掩码
        y_attn_mask = F.pad(  # yy的右上1扩展到左边xy的0,(y,x+y)
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool),
                       diagonal=1),  # 上三角矩阵,上三角部分为 True，表示音频序列不能看到未来的位置
            (x_len, 0),
            value=False,
        )  # [T_semantic, src_len],上三角矩阵在左侧（行方向）填充 x_len 列 False，即在音频序列的开头添加 x_len 列 False，使得音频序列也不能看到文本序列的位置
        xy_attn_mask = (
            torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
            .unsqueeze(0)
            .expand(bsz * self.num_head, -1, -1)
            .view(bsz, self.num_head, src_len, src_len)
            .to(device=x.device, dtype=torch.bool)
        )  # [1,self.num_head,src_len,src_len]，组合文本和音频的注意力掩码

        for idx in tqdm(range(1500)):
            if xy_attn_mask is not None:  # 现在看来第一次走这里
                # 处理文本和参考音频提示
                xy_dec, k_cache, v_cache = self.t2s_transformer.process_prompt(
                    xy_pos, xy_attn_mask, None)  # [1,src_len,512]. k_cache and v_cache are List[1,src_len,512]
            else:  # 第一次以后走这里
                # 解码下一个token,k{v}_cache是缓存
                xy_dec, k_cache, v_cache = self.t2s_transformer.decode_next_token(
                    xy_pos, k_cache, v_cache)  # 等号左侧的维度[1,1,512]。k_cache and v_cache are List[1,src_len+idx,512], 典型的LLM的decoder-only的形式

            # [1, 1025]。获取了上一步生成的文本和音频嵌入向量的最后一个时间步的输出。这里的xy_dec是前一步处理后的嵌入向量，结合了文本和音频信息。
            logits = self.ar_predict_layer(xy_dec[:, -1])

            if idx == 0:
                xy_attn_mask = None  # 这意味着在第一次预测时，模型不会使用任何之前的注意力掩码信息。
            if idx < 11:  # 至少预测出10个token不然不给停止（0.4s）
                logits = logits[:, :-1]  # 去掉最后一个

            samples = sample(
                logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature
            )[0]  # [1,1]。top_k和top_p用于控制采样的多样性，repetition_penalty用于惩罚重复生成的token，temperature用于控制生成的随机性。

            # [1,124+1=125], 将新的token拼到y最后，从而更新音频语义序列。y 的形状在每次循环后都会增加1
            y = torch.concat([y, samples], dim=1)

            # 提前停止,prefix_len参考音频token的长度,即在生成之前已经存在的token数。
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            # decode出EOS就停止。
            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                if y.shape[1] == 0:
                    # 如果为空，则拼接一个零token
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break

            ####################### update next step ###################################
            # [1,1,512],从y中提取最后一个token计算嵌入向量。
            y_emb = self.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[
                :, y_len + idx  # 缩放+位置编码权重*预计算好的位置编码矩阵（pe，postition encoding）
            ].to(dtype=y_emb.dtype, device=y_emb.device)  # [1,1,512]。这一步是为了让模型理解新生成的token在序列中的位置。

        if ref_free:
            return y[:, :-1], 0
        return y[:, :-1], idx

    def infer_panel(
        self,
        x: torch.LongTensor,  # 全部文本token
        x_lens: torch.LongTensor,
        prompts: torch.LongTensor,  # 参考音频token
        bert_feature: torch.LongTensor,  # 全部文本bert feature
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs,
    ):
        return self.infer_panel_naive(
            x, x_lens, prompts, bert_feature, top_k, top_p, early_stop_num, temperature, repetition_penalty, **kwargs
        )
