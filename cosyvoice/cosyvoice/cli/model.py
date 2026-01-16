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
                 llm: nn.Module,
                 flow: nn.Module,
                 hift: nn.Module,
                 fp16: bool = False):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16

        self.token_min_hop_len = self.flow.input_frame_rate * 2
        self.token_max_hop_len = self.flow.input_frame_rate * 4
        self.token_overlap_len = 20
        # mel fade in out
        self.mel_overlap_len = int(
            self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(256 * self.mel_cache_len)
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
