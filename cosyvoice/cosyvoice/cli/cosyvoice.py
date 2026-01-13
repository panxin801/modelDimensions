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
            configs = load_hyperpyyaml(fr)
        assert get_model_type(
            configs) == CosyVoice, f"do not use {model_dir} for CosyVoice initialization!"

        # Frontend and sr
        self.frontend = CosyVoiceFrontEnd(configs["get_tokenizer"],
                                          configs["feat_extractor"],
                                          f"{model_dir}/campplus.onnx",
                                          f"{model_dir}/speech_tokenizer_v1.onnx",
                                          f"{model_dir}/spk2info.pt",
                                          configs["allowed_special"])
        self.sample_rate = configs["sample_rate"]

        # Set flag
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            # No cuda device, then all set to false
            load_jit, load_trt, fp16 = False, False, False
            logging.warning(
                "No cuda device, set load_jit/load_trt/fp16 to False")

        # TTS model
        self.model = CosyVoiceModel(configs["llm"],
                                    configs["flow"],
                                    configs["hifi"],
                                    fp16)
        # load ckpt
        self.model.load(f"{model_dir}/llm.pt",
                        f"{model_dir}/flow.pt",
                        f"{model_dir}/hifi.pt")

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
