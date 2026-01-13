# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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

import json
import os
import re
import inflect
import onnxruntime
import torch
import numpy as np
import whisper
import torchaudio.compliance.kaldi as kaldi
from functools import partial
from typing import (Generator, Callable)

from cosyvoice.utils.file_utils import (logging, load_wav)
from cosyvoice.utils.frontend_utils import (
    contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph, is_only_punctuation)


class CosyVoiceFrontEnd:
    def __init__(self, get_tokenizer: Callable,  # text tokenizer
                 feat_extractor: Callable,  # speech feature extractor
                 campplus_mode: str,  # spk embedding extractor
                 speech_tokenizer_model: str,  # speech tokenizer model
                 spk2info: str = "",  # speaker info file
                 allowed_special: str = "all"):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.allowed_special = allowed_special
        self.inflect_parser = inflect.engine()  # Deal with english words

        # Onnx inference session
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_mode, sess_options=option, providers=["CPUExecutionProvider"])
        self.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option, providers=[
                                                                     "CPUExecutionProvider" if torch.cuda.is_available() else "CUDAExecutionProvider"])
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)
        else:
            self.spk2info = {}

        # NOTE compatible when no text frontend tool is avaliable
        try:
            import ttsfrd
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize(f"{ROOT_DIR}/../../pretrained_models/CosyVoice-ttsfrd/resource") is True, \
                "Failed to initialize ttsfrd resource"
            self.frd.set_lang_type("pinyinvg")
            self.text_frontend = "ttsfrd"
            logging.info("Use ttsfrd frontend")
        except:
            try:
                from wetext import Normalizer as ZhNormalizer
                from wetext import Normalizer as EnNormalizer
                self.zh_tn_model = ZhNormalizer(remove_erhua=False)
                self.en_tn_model = EnNormalizer()
                self.text_frontend = "wetext"
                logging.info("Use wetext frontend")
            except:
                self.text_frontend = ""
                logging.info("No frontend is avaliable")
