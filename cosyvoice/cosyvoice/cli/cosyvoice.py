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

import os
import torch
import time
from typing import Generator
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
# from cosyvoice.cli.model import (
#     CosyVoiceModel, CosyVoice2Model, CosyVoice3Model)
from cosyvoice.cli.model import (
    CosyVoiceModel,)
from cosyvoice.utils.file_utils import (logging)
from cosyvoice.utils.class_utils import (get_model_type)


class CosyVoice:
    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, trt_concurrent=1):
        self.model_dir = model_dir
        self.fp16 = fp16

        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)

        hyper_yaml_path = f"{model_dir}/cosyvoice.yaml"
        if not os.path.exists(hyper_yaml_path):
            raise ValueError(f"{hyper_yaml_path} not found!")
        with open(hyper_yaml_path, "rt", encoding="utf8") as fr:
            configs = load_hyperpyyaml(fr)  # 这里可以根据yaml文件里的class type直接实例化
        assert get_model_type(
            configs) == CosyVoiceModel, f"do not use {model_dir} for CosyVoice initialization!"

        # Frontend and sr
        # type(configs["get_tokenizer"]) is functools.partial
        # type(configs["parquet_opener"]) is function, because parquet_opener not has args in yaml file.
        self.frontend = CosyVoiceFrontEnd(configs["get_tokenizer"],
                                          configs["feat_extractor"],
                                          f"{model_dir}/campplus.onnx",
                                          f"{model_dir}/speech_tokenizer_v1.onnx",
                                          f"{model_dir}/spk2info.pt",
                                          configs["allowed_special"])  # in this func download some wetext model and fst stuff.
        self.sample_rate = configs["sample_rate"]  # 22050

        # Set flag
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            # No cuda device, then all set to false
            load_jit, load_trt, fp16 = False, False, False
            logging.warning(
                "No cuda device, set load_jit/load_trt/fp16 to False")

        # TTS model
        self.model = CosyVoiceModel(configs["llm"],
                                    configs["flow"],
                                    configs["hift"],
                                    fp16)  # False
        # load ckpt
        self.model.load(f"{model_dir}/llm.pt",
                        f"{model_dir}/flow.pt",
                        f"{model_dir}/hift.pt")

        # load jit
        if load_jit:
            self.model.load_jit(
                f"{model_dir}/llm.text_encoder.{'fp16' if self.fp16 else 'fp32'}.zip",
                f"{model_dir}/llm.llm.{'fp16' if self.fp16 else 'fp32'}.zip",
                f"{model_dir}/flow.encoder.{'fp16' if self.fp16 else 'fp32'}.zip")
        # load trt
        if load_trt:
            self.model.load_trt(f"{model_dir}/flow.decoder.estimator.{'fp16' if self.fp16 else 'fp32'}.mygpu.plan",
                                f"{model_dir}/flow.decoder.estiumator.fp32.onnx",
                                trt_concurrent,
                                self.fp16)

        del configs

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def add_zero_shot_spk(self,
                          prompt_text,
                          prompt_wav,
                          zero_shot_spk_id):
        assert zero_shot_spk_id != "", "do not use empty zero_shot_spk_id"
        model_input = self.frontend.frontend_zero_shot("",
                                                       prompt_text,
                                                       prompt_wav,
                                                       self.sample_rate,
                                                       "")
        del model_input["text"]
        del model_input["text_len"]
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        return True

    def save_spkinfo(self):
        torch.save(self.frontend.spk2info, f"{self.model_dir}/spk2info.pt")

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info(f"synthesis text {i}")
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output["tts_speech"].size(
                    1) / self.sample_rate
                logging.info(
                    f"yield speech len {speech_len}, rtf {(time.time()-start_time)/speech_len}")
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self,
                            tts_text,
                            prompt_text,
                            prompt_wav,
                            zero_shot_spk_id="",
                            stream=False,
                            speed=1.0,
                            text_frontend=True):
        if self.__class__.__name__ == "CosyVoice3" and "<|endofprompt|>" not in prompt_text + tts_text:
            logging.warning(
                "<|endofprompt|> not found in CosyVoice3 inference, check your input text")
        prompt_text = self.frontend.text_normalize(
            prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning(
                    f"synthesis text {i} too short than prompt text {prompt_text}, this may lead to bad performance")
            model_input = self.frontend.frontend_zero_shot(i, prompt_text,
                                                           prompt_wav,
                                                           self.sample_rate,
                                                           zero_shot_spk_id)
            start_time = time.time()
            logging.info(f"synthesis text {i}")
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output["tts_speech"].size(
                    1) / self.sample_rate
                logging.info(
                    f"yield speech len {speech_len}, rtf {(time.time()-start_time)/speech_len}")
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self,
                                tts_text,
                                prompt_wav,
                                zero_shot_spk_id="",
                                stream=False,
                                speed=1.0,
                                text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i,
                                                               prompt_wav,
                                                               self.sample_rate,
                                                               zero_shot_spk_id)
            start_time = time.time()
            logging.info(f"synthesis text {i}")
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output["tts_speech"].size(
                    1) / self.sample_rate
                logging.info(
                    f"yield speech len {speech_len}, rtf {(time.time()-start_time)/speech_len}")
                yield model_output
                start_time = time.time()

    def inference_vc(self,
                     source_wav,
                     prompt_wav,
                     stream=False,
                     speed=1.0):
        model_input = self.frontend.frontend_vc(source_wav,
                                                prompt_wav,
                                                self.sample_rate)
        start_time = time.time()
        for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
            speech_len = model_output["tts_speech"].size(1) / self.sample_rate
            logging.info(
                f"yield speech len {speech_len}, rtf {(time.time()-start_time)/speech_len}")
            yield model_output
            start_time = time.time()

    def inference_instruct(self,
                           tts_text,
                           spk_id,
                           instruct_text,
                           stream=False,
                           speed=1.0,
                           text_frontend=True):
        assert self.__class__.__name__ == "CosyVoice", "inference_instruct is only implemented for CosyVoice!"
        instruct_text = self.frontend.text_normalize(
            instruct_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(
                i, spk_id, instruct_text)
            start_time = time.time()
            logging.info(f"synthesis text {i}")
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output["tts_speech"].size(
                    1) / self.sample_rate
                logging.info(
                    f"yield speech len {speech_len}, rtf {(time.time()-start_time)/speech_len}")
                yield model_output
                start_time = time.time()


def AutoModel(**kwargs):
    if not os.path.exists(kwargs["model_dir"]):
        kwargs["model_dir"] = snapshot_download(kwargs["model_dir"])
    if os.path.exists(f"{kwargs['model_dir']}/cosyvoice.yaml"):
        return CosyVoice(**kwargs)
    elif os.path.exists(f"{kwargs['model_dir']}/cosyvoice2.yaml"):
        return CosyVoice2Model(**kwargs)
    elif os.path.exists(f"{kwargs['model_dir']}/cosyvoice3.yaml"):
        return CosyVoice3Model(**kwargs)
    else:
        raise TypeError("No valid model type found.")
