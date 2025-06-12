import torch
import json
import os

from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from process_ckpt import get_sovits_version_from_path_fast
from tools.i18n.i18n import I18nAuto


device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()

path_sovits_v3 = "GPT_SoVITS/pretrained_models/s2Gv3.pth"
path_sovits_v4 = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
is_exist_s2gv3 = os.path.exists(path_sovits_v3)
is_exist_s2gv4 = os.path.exists(path_sovits_v4)

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
    print("111")
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
    if prompt_language is not None and text_language is not None:
        if prompt_language in list(dict_language.keys()):
            prompt_text_update, prompt_language_update = (
                {"__type__": "update"},
                {"__type__": "update", "value": prompt_language},
            )
        else:
            prompt_text_update = {"__type__": "update", "value": ""}
            prompt_language_update = {
                "__type__": "update", "value": i18n("中文")}
        if text_language in list(dict_language.keys()):
            text_update, text_language_update = {"__type__": "update"}, {
                "__type__": "update", "value": text_language}
        else:
            text_update = {"__type__": "update", "value": ""}
            text_language_update = {"__type__": "update", "value": i18n("中文")}
        if model_version in v3v4set:
            visible_sample_steps = True
            visible_inp_refs = False
        else:
            visible_sample_steps = False
            visible_inp_refs = True
        yield (
            {"__type__": "update", "choices": list(dict_language.keys())},
            {"__type__": "update", "choices": list(dict_language.keys())},
            prompt_text_update,
            prompt_language_update,
            text_update,
            text_language_update,
            {"__type__": "update", "visible": visible_sample_steps, "value": 32 if model_version ==
                "v3"else 8, "choices": [4, 8, 16, 32, 64, 128]if model_version == "v3"else [4, 8, 16, 32]},
            {"__type__": "update", "visible": visible_inp_refs},
            {"__type__": "update", "value": False,
                "interactive": True if model_version not in v3v4set else False},
            {"__type__": "update", "visible": True if model_version == "v3" else False},
            {"__type__": "update", "value": i18n(
                "模型加载中，请等待"), "interactive": False},
        )


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
