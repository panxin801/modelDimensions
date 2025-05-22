import torch
import os
import sys
import traceback
import shutil
from time import time as ttime
from transformers import (AutoModelForMaskedLM, AutoTokenizer)

# Add path
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from tools import dataPrep_utils
from text.cleaner import clean_text


# inp_text=sys.argv[1]
# inp_wav_dir=sys.argv[2]
# exp_name=sys.argv[3]
# i_part=sys.argv[4]
# all_parts=sys.argv[5]
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[6]#i_gpu
# opt_dir="/data/docker/liujing04/gpt-vits/fine_tune_dataset/%s"%exp_name
# bert_pretrained_dir="/data/docker/liujing04/bert-vits2/Bert-VITS2-master20231106/bert/chinese-roberta-wwm-ext-large"


language_v1_to_language_v2 = {
    "ZH": "zh",
    "zh": "zh",
    "JP": "ja",
    "jp": "ja",
    "JA": "ja",
    "ja": "ja",
    "EN": "en",
    "en": "en",
    "En": "en",
    "KO": "ko",
    "Ko": "ko",
    "ko": "ko",
    "yue": "yue",
    "YUE": "yue",
    "Yue": "yue",
}


def my_save(fea, path):
    dir = os.path.dirname(path)
    name = os.path.basename(path)

    tmpPath = f"{ttime()}{i_part}.pth"
    torch.save(fea, tmpPath)
    shutil.move(tmpPath, os.path.join(dir, name))


def get_bert_feature(text, word2ph):
    with torch.inference_mode():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bertModel(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

    assert len(word2ph) == len(text)
    phoneLevelFeature = []
    for i in range(len(word2ph)):
        repeatFeature = res[i].repeat(word2ph[i], 1)
        phoneLevelFeature.append(repeatFeature)

    # [len(phones), 1024] 文本提取的bert特征。
    phoneLevelFeature = torch.cat(phoneLevelFeature, 0)
    return phoneLevelFeature.T


def process(data, res):
    for name, text, lan in data:
        try:
            name = dataPrep_utils.clean_path(name)
            name = os.path.basename(name)
            print(name)
            phones, word2ph, normText = clean_text(
                text.replace("%", "-").replace("￥", ","), lan, version)  # normText是正则化后文本，word2ph长度是字数，表示每个字在phones中占几个位置，sum(word2ph)=len(phones)
            pathBert = f"{bertDir}/{name}.pth"
            if not os.path.exists(pathBert) and lan == "zh":
                bertFeature = get_bert_feature(normText, word2ph)
                assert bertFeature.shape[-1] == len(phones)

                my_save(bertFeature, pathBert)
            phones = " ".join(phones)
            res.append([name, phones, word2ph, normText])
        except:
            print(name, text, traceback.format_exc())


if __name__ == "__main__":
    inp_text = os.environ.get("inp_text")
    inp_wav_dir = os.environ.get("inp_wav_dir")
    exp_name = os.environ.get("exp_name")
    i_part = os.environ.get("i_part")
    all_parts = os.environ.get("all_parts")
    if "_CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
    optDir = os.environ.get("opt_dir")
    bertPretrainDir = os.environ.get("bert_pretrained_dir")
    is_half = eval(os.environ.get("is_half", "True")
                   ) and torch.cuda.is_available()
    version = os.environ.get("version", None)

    txt_path = "%s/2-name2text-%s.txt" % (optDir, i_part)
    if not os.path.exists(txt_path):
        bertDir = f"{optDir}/3-bert"
        os.makedirs(optDir, exist_ok=True)
        os.makedirs(bertDir, exist_ok=True)
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        if not os.path.exists(bertPretrainDir):
            raise FileNotFoundError(bertPretrainDir)

        tokenizer = AutoTokenizer.from_pretrained(bertPretrainDir)
        bertModel = AutoModelForMaskedLM.from_pretrained(bertPretrainDir)
        if is_half:
            bertModel = bertModel.half().to(device)
        else:
            bertModel = bertModel.to(device)

        todo, res = [], []
        with open(inp_text, "rt", encoding="utf8") as fr:
            lines = fr.read().strip("\n").split("\n")

        for line in lines[int(i_part)::int(all_parts)]:
            try:
                wavName, spkName, language, text = line.split("|")
                if language in language_v1_to_language_v2.keys():
                    todo.append(
                        [wavName, text, language_v1_to_language_v2.get(language, language)])
                else:
                    print(
                        f"\033[33m[Waring] The {language = } of {wavName} is not supported for training.\033[0m")
            except:
                print(line, traceback.format_exc())

        process(todo, res)
        opt = []
        for name, phones, word2ph, normText in res:
            opt.append(f"{name}\t{phones}\t{word2ph}\t{normText}")
        with open(txt_path, "wt", encoding="utf8") as fw:
            fw.write("\n".join(opt) + "\n")
