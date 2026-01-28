# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
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

import torch
import os
import numpy as np
import time
import uuid
import threading
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Generator

from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import (
    convert_onnx_to_trt, export_cosyvoice2_vllm)
from cosyvoice.utils.common import TrtContextWrapper


class CosyVoiceModel:
    def __init__(self,
                 llm: nn.Module,  # TransformerLM
                 flow: nn.Module,  # MaskedDiffWithXVec
                 hift: nn.Module,  # HiFTGenerator
                 fp16: bool = False):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16

        self.token_min_hop_len = self.flow.input_frame_rate * 2  # 100
        self.token_max_hop_len = self.flow.input_frame_rate * 4  # 200
        self.token_overlap_len = 20
        # mel fade in out
        self.mel_overlap_len = int(
            self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)  # 34
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(256 * self.mel_cache_len)  # 5120
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)

        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, "Stream_scale_factor should be greater than 1, change it according to your actual rtf"
        self.llm_context = torch.cuda.stream(
            torch.cuda.Stream(device=self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()

        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}
        self.silent_tokens = []

    def load(self, llm_model,
             flow_model,
             hift_model,
             ):
        self.llm.load_state_dict(torch.load(
            llm_model, map_location=self.device, weights_only=True), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(
            flow_model, map_location=self.device, weights_only=True), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace("generator", ""): v for k, v in torch.load(
            hift_model, map_location=self.device, weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model,
                 llm_llm_model,
                 flow_encoder_model):
        llm_text_encoder = torch.jit.load(
            llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(
            flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self,
                 flow_decoder_estimator_model,
                 flow_decoder_onnx_model,
                 trt_concurrent,
                 fp16):
        assert torch.cuda.is_available(), "tensorrt only supports gpu!"
        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
            convert_onnx_to_trt(flow_decoder_estimator_model,
                                self.get_trt_kwargs(),
                                flow_decoder_onnx_model,
                                fp16)
        del self.flow.decoder.estimator

        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(
                trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, f"Failed to load trt {flow_decoder_estimator_model}"
        self.flow.decoder.estimator = TrtContextWrapper(
            estimator_engine, trt_concurrent=trt_concurrent, device=self.device)

    def get_trt_kwargs(self):
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def llm_job(self,
                text,
                prompt_text,
                llm_prompt_speech_token,
                llm_embedding,
                uuid):
        """llm_job 目前看来是用于TTS任务的函数

        :param text: target text token, [1, T_text]
        :param prompt_text: [1,0] 就是假的
        :param llm_prompt_speech_token: [1,0] 就是假的
        :param llm_embedding: spk embedding, [1, 192]
        :param uuid: str
        """
        cur_silent_token_num, max_silent_token_num = 0, 5
        with self.llm_context, torch.autocast("cuda", enabled=self.fp16 is True and hasattr(self.llm, "vllm") is False):
            # with self.llm_context, torch.cuda.amp.autocast(self.fp16 is True and hasattr(self.llm, 'vllm') is False):
            if isinstance(text, Generator):
                assert (self.__class__.__name__ !=
                        "CosyVoiceModel") and not hasattr(self.llm, "vllm"), f"streaming input text is only implemented for CosyVoice2/3 and do not support vllm!"
                token_generator = self.llm.inference_bistream(text=text,
                                                              prompt_text=prompt_text.to(
                                                                  self.device),
                                                              prompt_text_len=torch.tensor(
                                                                  [prompt_text.size(1)], dtype=torch.int).to(self.device),
                                                              prompt_speech_token=llm_prompt_speech_token.to(
                                                                  self.device),
                                                              prompt_speech_token_len=torch.tensor(
                                                                  [llm_prompt_speech_token.size(1)], dtype=torch.int).to(self.device),
                                                              embedding=llm_embedding.to(self.device),)
            else:
                token_generator = self.llm.inference(text=text.to(self.device),  # [1,T_text]
                                                     text_len=torch.tensor(
                                                         [text.size(1)], dtype=torch.int).to(self.device),  # [1]=T_text
                                                     prompt_text=prompt_text.to(
                                                         self.device),
                                                     prompt_text_len=torch.tensor(prompt_text.size(
                                                         1), dtype=torch.int).to(self.device),
                                                     prompt_speech_token=llm_prompt_speech_token.to(
                                                         self.device),
                                                     prompt_speech_token_len=torch.tensor(
                                                         [llm_prompt_speech_token.size(1)], dtype=torch.int).to(self.device),
                                                     embedding=llm_embedding.to(
                                                         self.device),  # [1,192]
                                                     uuid=uuid)
            for i in token_generator:
                if i in self.silent_tokens:
                    cur_silent_token_num += 1
                    if cur_silent_token_num > max_silent_token_num:
                        continue
                else:
                    cur_silent_token_num = 0
                self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def vc_job(self,
               source_speech_token,
               uuid):
        self.tts_speech_token_dict[uuid] = source_speech_token.flatten(
        ).tolist()
        self.llm_end_dict[uuid] = True

    def token2wav(self,
                  token,
                  prompt_token,
                  prompt_feat,
                  embedding,
                  uuid,
                  finalize=False,
                  speed=1.0):
        with torch.autocast("cuda", enabled=self.fp16):
            tts_mel, self.flow_cache_dict[uuid] = self.flow.inference(token=token.to(self.device, dtype=torch.int),
                                                                      token_len=torch.tensor(
                                                                          [token.size(1)], dtype=torch.int, device=self.device),
                                                                      prompt_token=prompt_token.to(
                                                                          self.device),
                                                                      prompt_token_len=torch.tensor(
                                                                          [prompt_token.size(1)], dtype=torch.int, device=self.device),
                                                                      prompt_feat=prompt_feat.to(
                                                                          self.device),
                                                                      prompt_feat_len=torch.tensor(
                                                                          [prompt_feat.size(1)], dtype=torch.int, device=self.device),
                                                                      embedding=embedding.to(
                                                                          self.device),
                                                                      flow_cache=self.flow_cache_dict[uuid],)

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid].size(2) != 0:
            tts_mel = fade_in_out(
                tts_mel,
                self.mel_overlap_dict[uuid],
                self.mel_window)
        # append hift cache
        if not self.hift_cache_dict[uuid] is None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[
                uuid]["mel"], self.hift_cache_dict[uuid]["source"]
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)  # !!!!!!
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:,
                                                  :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(
                speech_feat=tts_mel, cache_source=hift_cache_source)
            if not self.hift_cache_dict[uuid] is None:
                tts_speech = fade_in_out(tts_speech,
                                         self.hift_cache_dict[uuid]["speech"],
                                         self.speech_window)
            self.hift_cache_dict[uuid] = {"mel": tts_mel[:, :, -self.mel_cache_len:],
                                          "source": tts_source[:, :, -self.source_cache_len:],
                                          "speech": tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, f"speed change only support non - stream inference mode"
                tts_mel = F.interpolate(tts_mel,
                                        size=int(tts_mel.size(2) / speed),
                                        mode="linear")
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel,
                                                         cache_source=hift_cache_source)
            if not self.hift_cache_dict[uuid] is None:
                tts_speech = fade_in_out(tts_speech,
                                         self.hift_cache_dict[uuid]["speech"],
                                         self.speech_window)
        return tts_speech

    def tts(self,
            text=torch.zeros(1, 0, dtype=torch.int32),
            flow_embedding=torch.zeros(0, 192,),
            llm_embedding=torch.zeros(0, 192,),
            prompt_text=torch.zeros(1, 0, dtype=torch.int),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int),
            prompt_speech_feat=torch.zeros(1, 0, 80),
            source_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            stream=False,
            speed=1.0,
            **kwargs):
        """tts main entry point

        :param text: target generated text from whisper tokenizer, shape=[1, T_text], dtype=int(know as int32)
        :param flow_embedding: spk embedding from cam++, shape=[1,192], dtype=float32
        :param llm_embedding: spk embedding from cam++, shape=[1,192], dtype=float32
        :param prompt_text: 说明
        :param llm_prompt_speech_token: 说明
        :param flow_prompt_speech_token: 说明
        :param prompt_speech_feat: 说明
        :param source_speech_token: 说明
        :param stream: True for streaming inference
        :param speed: output speech speed
        :param kwargs: 说明

        Return
            generator: dict
        """

        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())  # str, uuid
        """self.lock 的作用是确保对共享资源的线程安全访问。具体来说，这段代码中使用 self.lock 来保护对以下字典的初始化和修改操作：
        self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
        self.hift_cache_dict[this_uuid] = None
        self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
        self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        没有锁 会有问题
        竞争条件（Race Condition），数据不一致
        """
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [
            ], False
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(
                1, 80, 0)  # [1,80,0]
            self.flow_cache_dict[this_uuid] = torch.zeros(
                1, 80, 0, 2)  # [1,80,0,2]
        if source_speech_token.size(1) == 0:
            # TTS 任务, 如下外层函数会使用inference_sft
            p = threading.Thread(target=self.llm_job,
                                 args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            # VC 任务, 如下外层函数会使用
            p = threading.Thread(target=self.vc_job,
                                 args=(source_speech_token, this_uuid))
        p.start()
        if stream is True:
            token_hop_len = self.token_min_hop_len  # 100
            while True:
                time.sleep(1e-1)
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(
                        self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]).unsqueeze(0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {"tts_speech": this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(
                        token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(
                self.tts_speech_token_dict[this_uuid]).unsqueeze(0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {"tts_speech": this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(
                self.tts_speech_token_dict[this_uuid]).unsqueeze(0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {"tts_speech": this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.flow_cache_dict.pop(this_uuid)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()
