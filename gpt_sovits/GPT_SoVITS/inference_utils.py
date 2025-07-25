import torch
import json
import os
import librosa
import traceback
import torchaudio
import re
import numpy as np
# parameter efficient fine tunget_peft_modeling
from peft import (LoraConfig, get_peft_model)
from time import time as ttime
from transformers import (AutoModelForMaskedLM, AutoTokenizer)

from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from process_ckpt import (get_sovits_version_from_path_fast, load_sovits_new)
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.module.models import (
    SynthesizerTrn, SynthesizerTrnV3, Generator)
from feature_extractor import cnhubert
from text import (chinese, cleaned_text_to_sequence)
from text.cleaner import clean_text
from text.LangSegmenter import LangSegmenter
from module.mel_processing import (mel_spectrogram_torch, spectrogram_torch)
from BigVGAN import bigvgan

device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}
punctuation = set(["!", "?", "…", ",", ".", "-", " "])
dtype = torch.float16 if is_half else torch.float32


path_sovits_v3 = "GPT_SoVITS/pretrained_models/s2Gv3.pth"
path_sovits_v4 = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
is_exist_s2gv3 = os.path.exists(path_sovits_v3)
is_exist_s2gv4 = os.path.exists(path_sovits_v4)
v3v4set = {"v3", "v4"}
version = model_version = os.environ.get("version", "v2")
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base")
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")

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
dict_language = dict_language_v1 if version == "v1" else dict_language_v2
spec_min = -12
spec_max = 2
sr_model = None

# SSL model
cnhubert.cnhubert_base_path = cnhubert_base_path
ssl_model = cnhubert.get_model()
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)

if is_half is True:
    ssl_model = ssl_model.half().to(device)
    bert_model = bert_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)
    bert_model = bert_model.to(device)

# ref_wav_path+prompt_text+prompt_language+text(单个)+text_language+top_k+top_p+temperature
# cache_tokens={}#暂未实现清理机制
cache = {}
resample_transform_dict = {}


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


if not os.path.exists("./weight.json"):
    with open("./weight.json", "wt", encoding="utf8") as fw:
        json.dump({"GPT": {}, "SoVITS": {}}, fw)


def audio_sr(audio, sr):
    global sr_model
    if sr_model is None:
        from tools.audio_sr import AP_BWE

        try:
            sr_model = AP_BWE(device, DictToAttrRecursive)
        except FileNotFoundError:
            print(i18n("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载好"))
            return audio.cpu().detach().numpy(), sr
    return sr_model(audio, sr)


def mel_fn(x):
    return mel_spectrogram_torch(
        x,
        **{
            "n_fft": 1024,
            "win_size": 1024,
            "hop_size": 256,
            "num_mels": 100,
            "sampling_rate": 24000,
            "fmin": 0,
            "fmax": None,
            "center": False,
        },
    )


def mel_fn_v4(x):
    return mel_spectrogram_torch(
        x,
        **{
            "n_fft": 1280,
            "win_size": 1280,
            "hop_size": 320,
            "num_mels": 100,
            "sampling_rate": 32000,
            "fmin": 0,
            "fmax": None,
            "center": False,
        },
    )


def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min


def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1


def resample(audio_tensor, sr0, sr1):
    global resample_transform_dict
    key = f"{sr0}-{sr1}"
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(
            sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  # 如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut4(inp):
    inp = inp.strip("\n")
    opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    inp = inp.strip("\n")
    punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


def process_text(texts):
    _text = []
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError(i18n("请输入有效文本"))
    for text in texts:
        if text in [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text


def clean_text_inf(text, language, version):
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text


def get_spepc(hps, filename):
    # audio = load_audio(filename, int(hps.data.sampling_rate))
    audio, sampling_rate = librosa.load(
        filename, sr=int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)  # .to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


def get_bert_feature(text, word2ph):
    with torch.inference_mode():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


def get_phones_and_bert(text, language, version, final=False):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "all_zh":
            if re.search(r"[A-Za-z]", formattext):
                formattext = re.sub(
                    r"[a-z]", lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext, "zh", version)
            else:
                phones, word2ph, norm_text = clean_text_inf(
                    formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(device)
        elif language == "all_yue" and re.search(r"[A-Za-z]", formattext):
            formattext = re.sub(
                r"[a-z]", lambda x: x.group(0).upper(), formattext)
            formattext = chinese.mix_text_normalize(formattext)
            return get_phones_and_bert(formattext, "yue", version)
        else:
            phones, word2ph, norm_text = clean_text_inf(
                formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist = []
        langlist = []
        if language == "auto":
            for tmp in LangSegmenter.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(
                textlist[i], lang, version)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)

    return phones, bert.to(dtype), norm_text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


def init_bigvgan():
    global bigvgan_model, hifigan_model

    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        f"{os.getcwd()}/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x",
        use_cuda_kernel=False,
    )  # if True, RuntimeError: Ninja is required to load C++ extensions
    # remove weight norm in the model and set to eval mode
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    if hifigan_model:
        hifigan_model = hifigan_model.cpu()
        hifigan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass
    if is_half == True:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)


def init_hifigan():
    global hifigan_model, bigvgan_model
    hifigan_model = Generator(
        initial_channel=100,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 6, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 12, 4, 4, 4],
        gin_channels=0, is_bias=True
    )
    hifigan_model.eval()
    hifigan_model.remove_weight_norm()
    state_dict_g = torch.load(
        f"{os.getcwd()}/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth", map_location="cpu")
    print("loading vocoder", hifigan_model.load_state_dict(state_dict_g))
    if bigvgan_model:
        bigvgan_model = bigvgan_model.cpu()
        bigvgan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass
    if is_half == True:
        hifigan_model = hifigan_model.half().to(device)
    else:
        hifigan_model = hifigan_model.to(device)


bigvgan_model = hifigan_model = None


def change_gpt_weights(gpt_path):
    global t2s_model, hz, max_sec
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
    global model_version, dict_language, version, hps, vq_model
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
            del vq_model.enc_q
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
        # W_lora=W_orig+(alpha/r)*delt_w
        lora_confg = LoraConfig(target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                                r=lora_rank,
                                lora_alpha=lora_rank,
                                init_lora_weights=True)
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_confg)
        print(f"Loading sovits_{model_version}_lora{lora_rank}")
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()  # 模型合并，base model和lora模型
        vq_model.eval()

    with open("./weight.json") as fr:
        data = fr.read()
        data = json.loads(data)
        data["SoVITS"][version] = sovits_path
    with open("./weight.json", "w") as fw:
        fw.write(json.dumps(data))


def get_tts_wav(
    ref_wav_path,
    prompt_text,  # 标注文本
    prompt_language,
    text,  # 生成文本
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
    global cache
    t = []
    if len(prompt_text) == 0 or prompt_text is None:
        ref_free = True
    if model_version in v3v4set:
        ref_free = False  # s2v3暂不支持ref_free
    else:
        if_sr = False

    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    if not ref_free:
        # 切分参考文本
        prompt_text = prompt_text.strip("\n")
        if not prompt_text[-1] in splits:
            prompt_text += "。" if prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)
    # 切分目标文本
    text = text.strip("\n")
    print(i18n("实际输入的目标文本:"), text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * pause_second),
        dtype=np.float16 if is_half is True else np.float32,
    )  # 空白音频
    zero_wav_torch = torch.from_numpy(zero_wav)
    if is_half is True:
        zero_wav_torch = zero_wav_torch.half().to(device)
    else:
        zero_wav_torch = zero_wav_torch.to(device)
    if not ref_free:
        with torch.inference_mode():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)  # 参考音频，按16k采样率读取
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                print(i18n("参考音频在3~10秒范围外，请更换！"))
                raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            wav16k = torch.from_numpy(wav16k)
            if is_half is True:
                wav16k = wav16k.half().to(device)
            else:
                wav16k = wav16k.to(device)
            # 参考音频后拼接空白音频, size=[80000]
            wav16k = torch.cat([wav16k, zero_wav_torch])

            # SSL
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"].transpose(1, 2)  # float16(), [B,768, T_semantic=249]，参考音频的ssl
            codes = vq_model.extract_latent(
                ssl_content)  # [B,1,floor(T_semantic/2)=124]，计算潜在的语义编码
            prompt_semantic = codes[0, 0]  # [floor(T_semantic/2)]
            prompt = prompt_semantic.unsqueeze(0).to(device)
            print(prompt.shape)  # [1,floor(T_semantic/2)]

    t1 = ttime()
    t.append(t1 - t0)

    # Split text
    if how_to_cut == i18n("凑四句一切"):
        text = cut1(text)
    elif how_to_cut == i18n("凑50字一切"):
        text = cut2(text)
    elif how_to_cut == i18n("按中文句号。切"):
        text = cut3(text)
    elif how_to_cut == i18n("按英文句号.切"):
        text = cut4(text)
    elif how_to_cut == i18n("按标点符号切"):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    print(i18n("实际输入的目标文本(切句后):"), text)
    texts = text.split("\n")
    texts = process_text(texts)  # 处理空白
    texts = merge_short_text_in_array(texts, 5)  # 合并短句后的TTS目标文本

    audio_opt = []
    # s2v3暂不支持ref_free
    if not ref_free:
        # phones1 is list, len=30, define as T_phoneme1. bert1 [1024, T_phoneme1]
        # 这里处理的是参考文本
        phones1, bert1, norm_text1 = get_phones_and_bert(
            prompt_text, prompt_language, version)

    for i_text, text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if len(text.strip()) == 0:
            continue

        if not text[-1] in splits:
            text += "。" if text_language != "en" else "."
        print(i18n("实际输入的目标文本(每句):"), text)
        # phones2 is list, len=11, define as T_phoneme2. bert2 [1024, T_phoneme2], len(norm_text2)=文本长度
        # 这里处理的是目标文本
        phones2, bert2, norm_text2 = get_phones_and_bert(
            text, text_language, version)
        print(i18n("前端处理后的文本(每句):"), norm_text2)

        if not ref_free:
            # [B, T_phoneme2+T_phoneme1], 拼接参考bert和目标bert
            bert = torch.cat([bert1, bert2], dim=1)
            all_phoneme_ids = torch.LongTensor(
                phones1 + phones2).to(device).unsqueeze(0)  # [1, T_phoneme2+T_phoneme1], 拼接参考phoneme_ids和目标phoneme_ids
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(dtype).unsqueeze(0)  # [B, 1, T_phoneme2+T_phoneme1]
        all_phoneme_len = torch.tensor(
            [all_phoneme_ids.shape[1]]).to(device)  # [1]=41,包含了全部音素token的张量

        t2 = ttime()
        if i_text in cache and if_freeze is True:
            # cache是每个文本片段i_text对应它的语义特征（pred_semantic）
            # 以便在处理相同的文本片段时可以直接从缓存中获取结果，而不需要重复进行计算
            pred_semantic = cache[i_text]
        else:
            with torch.inference_mode():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,  # 全部音素token包含了全部音素token的张量
                    all_phoneme_len,
                    None if ref_free else prompt,  # 参考音频SSL,过SoVITS计算得到的语义编码
                    bert,  # 全部文本Bert
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=hz * max_sec)
                # [1,1,idx=38], 最后idx个
                # 也就是文本片段i_text对应它的语义特征（pred_semantic）
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                cache[i_text] = pred_semantic
        t3 = ttime()

        # v3不存在以下逻辑和inp_refs
        if not model_version in v3v4set:
            refers = []
            if inp_refs:  # 用户提供了额外的参考音频
                for path in inp_refs:
                    try:
                        refer = get_spepc(hps, path.name).to(
                            device=device, dtype=dtype)  # 从参考音频路径中提取特征
                        refers.append(refer)
                    except:
                        traceback.print_exc()
            if len(refers) == 0:  # 没有额外的参考音频，使用参考音频
                refers = [get_spepc(hps, ref_wav_path).to(
                    device=device, dtype=dtype)]
            audio = vq_model.decode(pred_semantic, torch.LongTensor(
                phones2).to(device).unsqueeze(0), refers, speed=speed)[0][0]  # .cpu().detach().numpy()
        else:
            refer = get_spepc(hps, ref_wav_path).to(
                device).to(dtype)  # [1,1025, T_refwav=220]
            phoneme_ids0 = torch.LongTensor(phones1).to(
                device).unsqueeze(0)  # [1, T_phoneme1]
            phoneme_ids1 = torch.LongTensor(phones2).to(
                device).unsqueeze(0)  # [1, T_phoneme2]
            fea_ref, ge = vq_model.decode_encp(
                prompt.unsqueeze(0), phoneme_ids0, refer)  # [1,512,T=496],[1,512,1],prompt是参考音频的ssl过SoVITS的latent code，refer是参考音频的特征
            ref_audio, sr = torchaudio.load(ref_wav_path)
            ref_audio = ref_audio.to(device).float()
            if ref_audio.size(0) == 2:
                ref_audio = ref_audio.mean(0).unsqueeze(0)
            tgt_sr = 24000 if model_version == "v3" else 32000
            if sr != tgt_sr:
                ref_audio = resample(ref_audio, sr, tgt_sr)
            mel2 = mel_fn(
                ref_audio) if model_version == "v3" else mel_fn_v4(ref_audio)  # [1,100,T=440]
            mel2 = norm_spec(mel2)  # 归一化后的Mel spec。
            T_min = min(mel2.size(2), fea_ref.size(2))  # min(440,496)
            mel2 = mel2[:, :, :T_min]  # [1,100,440]
            fea_ref = fea_ref[:, :, :T_min]  # [1,512,440]
            Tref = 468 if model_version == "v3"else 500
            Tchunk = 934 if model_version == "v3"else 1000
            if T_min > Tref:
                mel2 = mel2[:, :, -Tref:]
                fea_ref = fea_ref[:, :, -Tref:]
                T_min = Tref
            chunk_len = Tchunk - T_min  # 560
            mel2 = mel2.to(dtype)
            fea_todo, ge = vq_model.decode_encp(
                pred_semantic, phoneme_ids1, refer, ge, speed)  # [1,512,136],[1,512,1]， pred_semantic目标文本语音文本,phoneme_ids1目标文本，refer是参考音频特征
            cfm_ress = []
            idx = 0
            while 1:
                # 不足chunk_len就读到末尾
                fea_todo_chunk = fea_todo[:, :,
                                          idx:idx + chunk_len]  # [1,512,136]
                if fea_todo_chunk.size(-1) == 0:
                    break
                idx += chunk_len
                fea = torch.cat([fea_ref, fea_todo_chunk],
                                2).transpose(2, 1)  # [1,572,512],将参考特征fea_ref和目标特征块fea_todo_chunk在第三个维度上进行拼接
                cfm_res = vq_model.cfm.inference(fea, torch.LongTensor([fea.size(1)]).to(
                    fea.device), mel2, sample_steps, inference_cfg_rate=0)  # [1,100,588]，对拼接后的特征fea进行处理，结合mel2特征进行特征融合与映射，生成当前特征块的中间结果cfm_res
                # [1,100,148]，取出新的Mel图特征部分，更新mel2用于下次循环
                cfm_res = cfm_res[:, :, mel2.size(2):]
                mel2 = cfm_res[:, :, -T_min:]  # [1,100,148]
                # 更新参考特征fea_ref，使其包含上一次处理的特征块的最后T_min个特征，为下一次循环做准备
                fea_ref = fea_todo_chunk[:, :, -T_min:]
                cfm_ress.append(cfm_res)  # [1,512,148]
            cfm_res = torch.cat(cfm_ress, 2)  # [1,100,148]
            cfm_res = denorm_spec(cfm_res)
            if model_version == "v3":
                if bigvgan_model is None:
                    init_bigvgan()
            else:  # v4
                if hifigan_model is None:
                    init_hifigan()
            vocoder_model = bigvgan_model if model_version == "v3" else hifigan_model
            with torch.inference_mode():
                wav_gen = vocoder_model(cfm_res)  # [1,1,71040],进入vocoder生成音频
                audio = wav_gen[0][0]  # .cpu().detach().numpy() size=[71040]

        max_audio = torch.abs(audio).max()  # 简单防止16bit爆音
        if max_audio > 1:
            audio = audio / max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav_torch)  # zero_wav 9600

        t4 = ttime()
        t.extend([t2 - t1, t3 - t2, t4 - t3])
        t1 = ttime()
    print(
        f"{t[0]:.3f}\t{sum(t[1::3]):.3f}\t{sum(t[2::3]):.3f}\t{sum(t[3::3]):.3f}")
    audio_opt = torch.cat(audio_opt, 0)
    if model_version in {"v1", "v2"}:
        opt_sr = 32000
    elif model_version == "v3":
        opt_sr = 24000
    else:
        opt_sr = 48000  # v4
    if if_sr is True and opt_sr == 24000:
        print(i18n("音频超分中"))
        audio_opt, opt_sr = audio_sr(audio_opt.unsqueeze(0), opt_sr)
        max_audio = np.abs(audio_opt).max()
        if max_audio > 1:
            audio_opt /= max_audio
    else:
        audio_opt = audio_opt.cpu().detach().numpy()
    yield opt_sr, (audio_opt * 32767).astype(np.int16)
