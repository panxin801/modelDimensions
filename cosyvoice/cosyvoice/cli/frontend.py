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

    def text_normalize(self, text, split=True, text_frontend=True):
        if isinstance(text, Generator):
            logging.info('get tts_text generator, will skip text_normalize!')
            return [text]
        # NOTE skip text_frontend when ssml symbol in text
        if '<|' in text and '|>' in text:
            text_frontend = False
        if text_frontend is False or text == '':
            return [text] if split is True else text
        text = text.strip()
        if self.text_frontend == 'ttsfrd':
            texts = [i["text"] for i in json.loads(
                self.frd.do_voicegen_frd(text))["sentences"]]
            text = ''.join(texts)
        else:
            if contains_chinese(text):
                if self.text_frontend == 'wetext':
                    text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r'[，,、]+$', '。', text)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
            else:
                if self.text_frontend == 'wetext':
                    text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text

    def _extract_text_token(self, text):
        if isinstance(text, Generator):
            logging.info(
                f"Get tts_text_generator, will return _extract_text_token_generator!")
            # NOTE add a dummy text_token_len for compatibility
            return self._extract_text_token_generator(text), torch.tensor([0], dtype=torch.int32).to(self.device)
        else:
            text_token = self.tokenizer.encode(
                text, allowed_special=self.allowed_special)
            text_token = torch.tensor(
                [text_token], dtype=torch.int32).to(self.device)  # torch.int32==torch.int
            text_token_len = torch.tensor(
                [text_token.size(1)], dtype=torch.int, device=self.device)
            return text_token, text_token_len

    def _extract_text_token_generator(self, text_generator):
        for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.size(1)):
                yield text_token[:, i:i + 1]

    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]["embedding"]
        model_input = {"text": tts_text_token,
                       "text_len": tts_text_token_len,
                       "llm_embedding": embedding,
                       "flow_embedding": embedding}
        # llm embedding and flow embedding are the same
        return model_input
