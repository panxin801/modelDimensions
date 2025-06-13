import torch
import json
import os
from peft import (LoraConfig, get_peft_model)

from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from process_ckpt import (get_sovits_version_from_path_fast, load_sovits_new)
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.module.models import (
    SynthesizerTrn, SynthesizerTrnV3, Generator)


device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()

path_sovits_v3 = "GPT_SoVITS/pretrained_models/s2Gv3.pth"
path_sovits_v4 = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
is_exist_s2gv3 = os.path.exists(path_sovits_v3)
is_exist_s2gv4 = os.path.exists(path_sovits_v4)
v3v4set = {"v3", "v4"}

i18n = I18nAuto()
dict_language_v1 = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
}
dict_language_v2 = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("粤语"): "all_yue",  # 全部按中文识别
    i18n("韩文"): "all_ko",  # 全部按韩文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("粤英混合"): "yue",  # 按粤英混合识别####不变
    i18n("韩英混合"): "ko",  # 按韩英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
    i18n("多语种混合(粤语)"): "auto_yue",  # 多语种启动切分识别语种
}


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


def change_gpt_weights(gpt_path):
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half is True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print(f"Number of parameter: {total/1e6:.2f}M")

    version = dict_s1["config"]["version"]
    with open("./weight.json") as fr:
        data = fr.read()
        data = json.loads(data)
        data["GPT"][version] = gpt_path
    with open("./weight.json", "w") as fw:
        fw.write(json.dumps(data))


def change_sovits_weights(sovits_path, prompt_language=None, text_language=None):
    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(
        sovits_path)
    print(sovits_path, version, model_version, if_lora_v3)
    is_exist = is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4

    if if_lora_v3 is True and is_exist is False:
        info = "GPT_SoVITS/pretrained_models/s2Gv3.pth" + \
            i18n("SoVITS %s 底模缺失，无法加载相应 LoRA 权重" % model_version)
        print(info)
        raise FileExistsError(info)

    dict_language = dict_language_v1 if version == "v1" else dict_language_v2

    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if not "enc_p.text_embedding.weight" in dict_s2["weight"]:
        hps.model.version = "v2"  # v3model,v2sybomls
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version = hps.model.version
    print("sovits版本:", hps.model.version)

    if not model_version in v3v4set:
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
        model_version = version
    else:
        hps.model.version = model_version
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )

    if not "pretrained" in sovits_path:
        try:
            del vq_model.enc_p
        except:
            pass
    if is_half is True:
        vq_model = vq_model.half()
    vq_model = vq_model.to(device)
    vq_model.eval()

    if if_lora_v3 is False:
        print(f"Loading sovits_{model_version}", vq_model.load_state_dict(
            dict_s2["weight"], strict=False))
    else:
        path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
        print(f"Loading sovits_{model_version}pretrained_G", vq_model.load_state_dict(
            load_sovits_new(path_sovits)["weight"], strict=False))
        lora_rank = dict_s2["lora_rank"]
        lora_confg = LoraConfig(target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                                r=lora_rank,
                                lora_alpha=lora_rank,
                                init_lora_weights=True)
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_confg)
        print(f"Loading sovits_{model_version}_lora{lora_rank}")
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()
        vq_model.eval()

    with open("./weight.json") as fr:
        data = fr.read()
        data = json.loads(data)
        data["SoVITS"][version] = sovits_path
    with open("./weight.json", "w") as fw:
        fw.write(json.dumps(data))


def get_tts_wav(
    ref_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    how_to_cut=i18n("不切"),
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    ref_free=False,
    speed=1,
    if_freeze=False,
    inp_refs=None,
    sample_steps=8,
    if_sr=False,
    pause_second=0.3,
):
    return None
