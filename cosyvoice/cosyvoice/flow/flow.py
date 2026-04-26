# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
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
import logging
import random
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from typing import (Dict, Optional)
from omegaconf import DictConfig

from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.utils.onnx import (
    onnx_path, online_feature, SpeechTokenExtractor)


class MaskedDiffWithXvec(nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 encoder: torch.nn.Module = None,  # ConformerEncoder
                 length_regulator: torch.nn.Module = None,  # InterpolateRegulator
                 decoder: torch.nn.Module = None,  # ConditionalCFM
                 decoder_conf: Dict = {'in_channels': 240,
                                       'out_channel': 80,
                                       'spk_emb_dim': 80,
                                       'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06,
                                                                 'solver': 'euler',
                                                                 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2,
                                                                 'inference_cfg_rate': 0.7,
                                                                 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256],
                                                          'dropout': 0.0,
                                                          'attention_head_dim': 64,
                                                          'n_blocks': 4,
                                                          'num_mid_blocks': 12,
                                                          'num_heads': 8,
                                                          'act_fn': 'gelu'}
                                       }):
        super().__init__()

        self.input_size = input_size  # 512
        self.output_size = output_size  # 80
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size  # 4096
        self.output_type = output_type  # "mel"
        self.input_frame_rate = input_frame_rate  # 50
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)  # 4096,512
        self.spk_embed_affine_layer = nn.Linear(
            spk_embed_dim, output_size)  # 192,80
        self.encoder = encoder  # ConformerEncoder
        self.encoder_proj = nn.Linear(
            self.encoder.output_size(), output_size)  # 512,80
        self.decoder = decoder  # ConditionalCFM
        self.length_regulator = length_regulator  # InterpolateRegulator
        self.only_mask_loss = only_mask_loss  # True

    def forward(self,
                batch: dict,
                device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        token = batch["speech_token"].to(device)
        token_len = batch["speech_token_len"].to(device)
        feat = batch["speech_feat"].to(device)
        feat_len = batch["speech_feat_len"].to(device)
        embedding = batch["embedding"].to(device)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, feat_len)

        # get conditions
        conds = torch.zeros(feat.size(), device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask(~make_pad_mask(feat_len)).to(h)
        # NOTE this is unnecessary, feat/h already same shape
        loss, _ = self.decoder.compute_loss(feat.transpose(1, 2).contiguous(),
                                            mask.unsqueeze(1),
                                            h.transpose(1, 2).contiguous(),
                                            embedding,
                                            cond=conds)
        return {"loss": loss}

    @torch.inference_mode()
    def inference(self,
                  token,  # [1,T_token] generated token from LLM
                  token_len,  # [1]=T_token
                  # [1,0], prompt speech token, [1, T_prompt_token=174]
                  prompt_token,
                  prompt_token_len,  # [1]=0
                  # [1,0,80]=0, prompt speech mel_spec, [1, T_prompt_mel=326, 80]
                  prompt_feat,
                  prompt_feat_len,  # [1]=0
                  embedding,  # [1,192], spk embedding
                  flow_cache):  # [1,80,0,2] all 0 is fake
        assert token.size(0) == 1
        # xvec projection, from cam++
        embedding = F.normalize(embedding, dim=1)  # [1,192]
        embedding = self.spk_embed_affine_layer(
            embedding)  # [1,80], v in fig1.C

        # concat prompt token and generated token, in semantic token level
        token_len1, token_len2 = prompt_token.size(
            1), token.size(1)  # 0,120. 174,120
        token, token_len = torch.concat(
            [prompt_token, token], dim=1), prompt_token_len + token_len
        # mask是semantic token level的
        mask = (~make_pad_mask(token_len)
                ).unsqueeze(-1).to(embedding)  # [1,120,1], float， make_pad_mask 本身是0 not pad，但是这里取反就是0（False）是pad。
        # [1,T_token,512], figure1.b 中llm 产生的token过的embedding layer
        token = self.input_embedding(torch.clamp(
            token, min=0)) * mask  # 乘mask去掉占位值的影响。

        # 从这里开始可以认为是Fig1.C的部分
        # semantic token encode
        # [B,T_token,512],[1,1,T_token]， token经过ConformerEncoder
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)  # [B,T_token,80]

        # 将h从semantic token level 扩展到mel_spec level, T_token -> T_mel
        # 由于semantic token是prompt_token|llm generated token的。因此扩展到mel_spec维度也是prompt_mel|llm generated mel的长度格式。
        mel_len1, mel_len2 = prompt_feat.shape[1], int(
            token_len2 / self.input_frame_rate * 22050 / 256)  # 0, 206. 326,206
        # 使用 length_regulator 对编码后的中间表示 h 进行长度调节，使其在时间维度上与目标梅尔频谱长度匹配。
        # h[:, :token_len1] 扩展到mel_len1长度，h[:, token_len1:] 扩展到mel_len2长度
        h, h_lengths = self.length_regulator.inference(
            h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)  # [1, mel_len1+mel_len2, 80], mel_len1+mel_len2=206

        # get conditions
        conds = torch.zeros(
            [1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)  # [B,T_mel,D]=[1, mel_len1+mel_len2=206, 80]
        conds[:, :mel_len1] = prompt_feat
        # conds=prompt_feat后边是mel_len2长度都是0。
        conds = conds.transpose(1, 2)  # [B,D,T_mel]=[1, 80,mel_len1+mel_len2]

        # [1, mel_len1+mel_len2]
        mask = (~make_pad_mask(torch.tensor(
            [mel_len1 + mel_len2]))).to(h)  # 1 意味着没pad
        # next lines corresond to the  Fig1.C in the paper
        # conditions formed like <v|mu|mask_speech_feat>
        # v is speaker embedding, here is embedding
        # mu is semantic tokens, concat prompt_token and llm generated token, here is h
        # mask_speech_feat is masked speech mel_spec, here is conds
        # x_t is intermediate state at timestep t. The noise in the self.decoder
        feat, flow_cache = self.decoder(mu=h.transpose(1, 2).contiguous(),  # [B,D, T_mel]
                                        mask=mask.unsqueeze(1),  # [B,1,T_mel]
                                        spks=embedding,  # [B，80]
                                        cond=conds,  # [B,D,T_mel]
                                        n_timesteps=10,
                                        prompt_len=mel_len1,  # int
                                        cache=flow_cache)  # return flow_cache=[1,80,34,2]=[B,D,prompt_len+34(mel_overlap_len),2],2 =z_cache and mu_cache
        feat = feat[:, :, mel_len1:]  # [1,80,206]， 取的prompt_feat拼接后的部分。
        assert feat.shape[2] == mel_len2
        return feat.float(), flow_cache


class CausalMaskedDiffWithXvec(nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 encoder: nn.Module = None,
                 decoder: nn.Module = None,
                 decoder_conf: Dict = {"in_channels": 240,
                                       "out_channel": 80,
                                       "spk_embed_dim": 80,
                                       "n_spks": 1,
                                       "cfm_params": DictConfig({"sigma_min": 1e-06,
                                                                 "solver": "euler",
                                                                "t_scheduler": "cosine",
                                                                 "training_cfg_rate": 0.2,
                                                                 "inference_cfg_rate": 0.7,
                                                                 "reg_loss_type": "l1"}),
                                       "decoder_params": {"channels": [256, 256],
                                                          "dropout": 0.0,
                                                          "attention_head_dim": 64,
                                                          "n_blocks": 4,
                                                          "num_mid_blocks": 12,
                                                          "num_heads": 8,
                                                          "act_fn": "gelu"}}):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len
        if online_feature is True:
            self.speech_token_extractor = SpeechTokenExtractor(
                os.path.join(onnx_path, "speech_tokenizer_v2.batch.onnx"))

    def forward(self,
                batch: dict,
                device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        if "speech_token" not in batch:
            token, token_len = self.speech_token_extractor.inference(
                batch["whisper_feat"], batch["whisper_feat_len"], device)
        else:
            token = batch["speech_token"].to(device)
            token_len = batch["speech_token_len"].to(device)
        feat = batch["speech_feat"].to(device)
        feat_len = batch["speech_feat_len"].to(device)
        embedding = batch["embedding"].to(device)

        # NOTE unified training, static_chunk_size > 0 or = 0
        streaming = True if random.random() < 0.5 else False

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, 0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len, streaming=streaming)
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(h_lengths.sum(-1).squeeze(1))).to(h)
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
            streaming=streaming,)
        return {"loss": loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  streaming,
                  finalize):
        assert token.size(0) == 1
        # xvec projection
        embedding = F.normalize(embedding, 1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token, token_len = torch.concat(
            [prompt_token, token], 1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, 0)) * mask

        # text encode
        if finalize is True:
            h, h_lengths = self.encoder(token, token_len, streaming=streaming)
        else:
            token, context = token[:, :-
                                   self.pre_lookahead_len], token[:, -self.pre_lookahead_len:]
            h, h_lengths = self.encoder(
                token, token_len, context=context, streaming=streaming)
        mel_len1, mel_len2 = prompt_feat.size(
            1), h.size(1) - prompt_feat.size(1)
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros(
            [1, mel_len1 + mel_len2, self.output_size], device=token.device, dtype=h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming)
        feat = feat[:, :, mel_len1:]
        assert feat.size(2) == mel_len2
        return feat.float(), None


class CausalMaskedDiffWithDiT(nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 pre_lookahead_layer: nn.Module = None,
                 decoder: nn.Module = None,
                 decoder_conf: Dict = {"in_channels": 240,
                                       "out_channel": 80,
                                       "spk_embed_dim": 80,
                                       "n_spks": 1,
                                       "cfm_params": DictConfig({"sigma_min": 1e-06,
                                                                 "solver": "euler",
                                                                "t_scheduler": "cosine",
                                                                 "training_cfg_rate": 0.2,
                                                                 "inference_cfg_rate": 0.7,
                                                                 "reg_loss_type": "l1"}),
                                       "decoder_params": {"channels": [256, 256],
                                                          "dropout": 0.0,
                                                          "attention_head_dim": 64,
                                                          "n_blocks": 4,
                                                          "num_mid_blocks": 12,
                                                          "num_heads": 8,
                                                          "act_fn": "gelu"}}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.pre_lookahead_len = pre_lookahead_len
        self.pre_lookahead_layer = pre_lookahead_layer
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio  # 2
        if online_feature is True:
            self.speech_token_extractor = SpeechTokenExtractor(
                model_path=os.path.join(onnx_path, "speech_tokenizer_v3.batch.onnx"))

    def forward(self,
                batch: dict,
                device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        if "speech_toke" not in batch:
            token, token_len = self.speech_token_extractor.inference(batch["whisper_feat"],
                                                                     batch["whisper_feat_len"],
                                                                     device)
        else:
            token = batch["speech_token"].to(device)
            token_len = batch["speech_token_len"].to(device)
        feat = batch["speech_feat"].to(device)
        feat_len = batch["speech_feat_len"].to(device)
        embedding = batch["embedding"].to(device)

        # NOTE unified training, static_chunk_size > 0 or = 0
        streaming = True if random.random() < 0.5 else False

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embeding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, 0)) * mask

        # text encode
        h = self.pre_lookahead_layer(token)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        mask = mask.repeat_interleave(self.token_mel_ratio, dim=1).squeeze(-1)

        # get conditions
        conds = torch.zeros(feat.size(), device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
            streaming=streaming,
        )
        return {"loss": loss}

    @torch.inference_mode()
    def inference(self,
                  token,  # LLM generated token, [B, T_generated_token=210]
                  token_len,
                  # prompt speech token, [B, T_prompt_speech_token=87]
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,  # prompt mel spec, [B, T_prompt_mel=174, D=80]
                  prompt_feat_len,
                  embedding,  # spk embedding, [B, D=192]
                  streaming,
                  finalize):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, 1)  # [B, 192]
        embedding = self.spk_embed_affine_layer(embedding)  # [B, 80]

        # concat text and prompt_text
        token, token_len = torch.concat(
            [prompt_token, token], 1), prompt_token_len + token_len  # [B, T_token=T_prompt_speech_token+T_generated_token=297]
        mask = (~make_pad_mask(token_len)
                ).unsqueeze(-1).to(embedding)  # [B, T_token, 1]
        token = self.input_embedding(torch.clamp(
            token, 0)) * mask  # [B, T_token, 80]

        # text encode
        if finalize is True:
            h = self.pre_lookahead_layer(token)  # [B, T_token, 80]
        else:
            h = self.pre_lookahead_layer(
                token[:, :-self.pre_lookahead_len], context=token[:, -self.pre_lookahead_len:])
        h = h.repeat_interleave(self.token_mel_ratio, 1)  # [B, 2*T_token, 80]
        mel_len1, mel_len2 = prompt_feat.size(
            1), h.size(1) - prompt_feat.size(1)  # T_prompt_mel, 2*T_token-T_prompt_mel

        # get conditions
        conds = torch.zeros(
            [1, mel_len1 + mel_len2, self.output_size], device=token.device, dtype=h.dtype)  # [B, 2*T_token, 80]
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)  # [B, 80, 2*T_token]

        # [B, 2*T_token]
        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(mu=h.transpose(1, 2).contiguous(),  # semantic token hidden state [B, 80, 2*T_token]
                               mask=mask.unsqueeze(1),
                               spks=embedding,  # [B, 80]
                               cond=conds,  # [B, 80, 2*T_token], mel spec
                               n_timesteps=10,
                               streaming=streaming)  # [B, 80, 2*T_token=594]
        feat = feat[:, :, mel_len1:]  # [B, 80, mel_len2]
        assert feat.size(2) == mel_len2
        return feat.float(), None


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from hyperpyyaml import load_hyperpyyaml
    with open('./pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml', 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'hift': None})
    model = configs['flow']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    max_len = 10 * model.decoder.estimator.static_chunk_size
    chunk_size = model.decoder.estimator.static_chunk_size
    context_size = model.pre_lookahead_layer.pre_lookahead_len
    token = torch.randint(0, 6561, size=(1, max_len)).to(device)
    token_len = torch.tensor([max_len]).to(device)
    prompt_token = torch.randint(0, 6561, size=(1, chunk_size)).to(device)
    prompt_token_len = torch.tensor([chunk_size]).to(device)
    prompt_feat = torch.rand(1, chunk_size * 2, 80).to(device)
    prompt_feat_len = torch.tensor([chunk_size * 2]).to(device)
    prompt_embedding = torch.rand(1, 192).to(device)
    pred_gt, _ = model.inference(token, token_len, prompt_token, prompt_token_len,
                                 prompt_feat, prompt_feat_len, prompt_embedding, streaming=True, finalize=True)
    for i in range(0, max_len, chunk_size):
        finalize = True if i + chunk_size + context_size >= max_len else False
        pred_chunk, _ = model.inference(token[:, :i + chunk_size + context_size], torch.tensor([token[:, :i + chunk_size + context_size].shape[1]]).to(device),
                                        prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, prompt_embedding, streaming=True, finalize=finalize)
        pred_chunk = pred_chunk[:, :, i * model.token_mel_ratio:]
        print((pred_gt[:, :, i * model.token_mel_ratio: i * model.token_mel_ratio +
              pred_chunk.shape[2]] - pred_chunk).abs().max().item())
