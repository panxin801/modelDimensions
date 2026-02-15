# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li, Qihua, Shengqiang Li)
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
import numpy as np
from typing import (Dict, Optional, Callable, List, Generator)
from torch.nn.utils.rnn import (pad_sequence, unpad_sequence)

from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import (IGNORE_ID, th_accuracy)
from cosyvoice.utils.file_utils import logging


class TransformerLM(nn.Module):
    def __init__(self,
                 text_encoder_input_size: int,  # 512
                 llm_input_size: int,  # 1024
                 llm_output_size: int,  # 1024
                 text_token_size: int,  # 51866
                 speech_token_size: int,  # 4096
                 text_encoder: torch.nn.Module,  # ConformerEncoder
                 llm: torch.nn.Module,  # TransformerEncoder
                 sampling: Callable,  # ras_sampling
                 length_normalized_loss: bool = True,  # True
                 lsm_weight: float = 0.0,  # 0
                 spk_embed_dim: int = 192,):  # 192
        super().__init__()

        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = nn.Embedding(
            text_token_size, text_encoder_input_size)  # 51866,512
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(), llm_input_size)  # 1024,1024

        # 2. build speech token language model related modules
        self.sos = 0
        self.task_id = 1
        self.eos_token = self.speech_token_size  # 4096
        # llm_input_size=1024,self.llm_embedding.weight.size()=[2,1024], used for sos symbol and task id
        self.llm_embedding = nn.Embedding(2, llm_input_size)
        self.llm = llm  # TransformerEncoder
        self.llm_decoder = nn.Linear(
            llm_output_size, speech_token_size + 1)  # 1024,4096+1
        self.criterion_ce = LabelSmoothingLoss(size=speech_token_size + 1,  # 4096+1
                                               padding_idx=IGNORE_ID,  # -1
                                               smoothing=lsm_weight,  # 0
                                               normalize_length=length_normalized_loss,)  # True

        # 3. [Optional] build speech token related modules
        self.speech_embedding = nn.Embedding(
            speech_token_size, llm_input_size)  # 4096,1024
        self.spk_embed_affine_layer = nn.Linear(
            spk_embed_dim, llm_input_size)  # 192, 1024

        # 4. sampling method
        self.sampling = sampling

    def encode(self, text: torch.Tensor,
               text_lengths: torch.Tensor):
        """ encode text embeddings
        Args:
            text: [1, T_text, 512]
            text_lengths: [1]
        Return:
            encoder_out:  [1,T_text,1024]
            encoder_out_lens: [1]=T_text
        """
        encoder_out, encoder_mask = self.text_encoder(text,
                                                      text_lengths,
                                                      decoding_chunk_size=1,
                                                      num_decoding_left_chunks=-1)  # [1,T_text,1024], [1,1,T_text]
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)  # [1]=T_text
        encoder_out = self.text_encoder_affine_layer(
            encoder_out)  # [1,T_text,1024]
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_emb,
                           embedding,
                           text_token,
                           text_token_len,
                           task_id_emb,
                           speech_token,
                           speech_token_len):
        text_token = unpad_sequence(
            text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(
            speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0)
                                    for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(
            lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(self,
                batch: dict,
                device: torch.device,
                ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch["text_token"].to(device)
        text_token_len = batch["text_token_len"].to(device)
        speech_token = batch["speech_token"].to(device)
        speech_token_len = batch["speech_token_len"].to(device)
        embedding = batch["embedding"].to(device)

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]
                                                                                       ].tolist() + [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(
            lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=-1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. sos and task_id
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(
            sos_emb,
            embedding,
            text_token,
            text_token_len,
            task_id_emb,
            speech_token,
            speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(
            logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {"loss": loss, "acc": acc}

    def sampling_ids(self,
                     weighted_scores: torch.Tensor,
                     decoded_tokens: List,
                     sampling: int,
                     ignore_eos: bool = True):
        """sampling_ids 的 Docstring
        :param weighted_scores: [4097] token generated from LLM, current token
        :param decoded_tokens:  already generated tokens, list
        :param sampling: 25
        :param ignore_eos:  if ignore EOS symbol, bool
        Return:
            top_ids: int id
        """
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(
                weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (top_ids < self.speech_token_size):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError(
                    f'sampling reaches max_trials {max_trials} and still get eos when ignore_eos is True, check your input!')
        return top_ids

    @torch.inference_mode()
    def inference(self,
                  text: torch.Tensor,
                  text_len: torch.Tensor,
                  prompt_text: torch.Tensor,
                  prompt_text_len: torch.Tensor,
                  prompt_speech_token: torch.Tensor,
                  prompt_speech_token_len: torch.Tensor,
                  embedding: torch.Tensor,
                  sampling: int = 25,
                  max_token_text_ratio: float = 20,
                  min_token_text_ratio: float = 2,
                  uuid: str = "",
                  ) -> Generator[torch.Tensor, None, None]:
        """ inference of TransformerLM, used for 

        :param text: target text token, [1, T_text]
        :param text_len: [1]=T_text
        :param prompt_text: [1,0] is fake. [B,T_prompt_text=16]
        :param prompt_text_len: [1]=0 is fake. [B]
        :param prompt_speech_token: [1,0] is fake. [B,T_prompt_speech_token=174]
        :param prompt_speech_token_len: [1]=0 is fake. [B]
        :param embedding: target spk embedding, [1,192]
        :param sampling: 采样用的参数
        :param max_token_text_ratio: 最高llm 产生token数量
        :param min_token_text_ratio: 最低llm 产生token数量
        :param uuid: uuid string

        Return:
            Generator[]
        """
        device = text.device
        # [1, T_prompt_text+T_text], 拼接了prompt_text and target text
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        # [ 1, T_text, 512], text token -> text embedding
        text = self.text_embedding(text)

        # 1. encode text, from text embedding space to llm space,
        # difference is llm space contains text token and speech token
        # [1, T_text, 1024], [1]=T_text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if embedding.size(0) != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)  # [1,1024]
            embedding = embedding.unsqueeze(1)  # [1,1,1024]
        else:
            embedding = torch.zeros(
                1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input, like paper figure 1
        sos_emb = self.llm_embedding.weight[self.sos].reshape(
            1, 1, -1)  # from [1024] -> [1,1,1024]
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(
            1, 1, -1)  # from [1024] -> [1,1,1024]
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(
                prompt_speech_token)  # [B, T_prompt_speech_token, 1024], speech token -> speech_embedding
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, self.llm_input_size, dtype=text.dtype).to(device)  # [1,0,1024] is fake
        lm_input = torch.concat(
            [sos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)
        # inference_sft: [1,36,1024], 36=1+1+33+1+0
        # zero_shot: [1,256,1024], 258=1+1+81+1+174

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len)
                      * min_token_text_ratio)  # 33*2, (81-16)*2
        max_len = int((text_len - prompt_text_len) *
                      max_token_text_ratio)  # 33*20, (81-16)*20

        # 5. step by step decode
        out_tokens = []  # change through decode step, save generated token
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros(
            (0, 0, 0, 0), device=lm_input.device)  # change through decode step, save cache for each step
        # att_cache is [0,0,0,0] at the beginning, cnn_cache is [0,0,0,0] at the beginning
        # input llm input to TransformerEncoder, per token generation, and then pass through a linear layer
        # treat as a classification task
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(
                # [1, T_text, 1024] for i=0, [1,1,1024] for i=1,2,3,4,...
                lm_input,
                # 0 for i=0, T_text+i-1 for i=1,2,3, change every step...
                offset=offset,
                required_cache_size=-1,
                # [0,0,0,0] for i=0, [elayer=14,16,T_text+i-1,128] for i=1,2,3..., change every step...
                att_cache=att_cache,
                # [0,0,0,0] for i=0, [elayer=14,0,0,0] for i=1,2,3,4,5...
                cnn_cache=cnn_cache,
                # [1,T_text,T_text] for i=0, [1,1,1] for i=1,2,3,4,5....
                att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                               device=lm_input.device)).to(torch.bool)
            )
            # Return shape:
            # [1,T_text,1024], [14,16,T_text+i,128], [14,0,0,0] for i=0,
            # [1,1,1024], [14,16,T_text+i,128], [14,0,0,0] for i=1,2,3...
            # ......

            # [1, speech_token_size+1]
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(
                logp.squeeze(dim=0),  # [4097]
                # [] for i=0, [token1,] for i=1,change every step...
                out_tokens,
                sampling,  # 25
                ignore_eos=True if i < min_len else False)
            if top_ids == self.eos_token:
                break  # 运行到这里退出函数
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)  # all newest token to out_tokens
            offset += lm_input.size(1)  # offset=T_text+i, change every step...

            lm_input = self.speech_embedding.weight[top_ids].reshape(
                1, 1, -1)  # [1,1,1024]
