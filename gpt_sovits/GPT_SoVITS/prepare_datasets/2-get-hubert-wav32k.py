import sys
import os
import torch
import traceback
import shutil
import numpy as np
import librosa
from scipy.io import wavfile
from time import time as ttime

# Add path
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from feature_extractor import cnhubert
from tools import dataPrep_utils

# from config import cnhubert_base_path
# cnhubert.cnhubert_base_path=cnhubert_base_path
# inp_text=sys.argv[1]
# inp_wav_dir=sys.argv[2]
# exp_name=sys.argv[3]
# i_part=sys.argv[4]
# all_parts=sys.argv[5]
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[6]
# cnhubert.cnhubert_base_path=sys.argv[7]
# opt_dir="/data/docker/liujing04/gpt-vits/fine_tune_dataset/%s"%exp_name


def my_save(fea, path):
    dir = os.path.dirname(path)
    name = os.path.basename(path)

    tmpPath = f"{ttime()}{i_part}.pth"
    torch.save(fea, tmpPath)
    os.makedirs(dir, exist_ok=True)
    shutil.move(tmpPath, os.path.join(dir, name))


def name2go(wav_name, wav_path):
    hubert_path = f"{hubert_dir}/{wav_name}.pt"
    if os.path.exists(hubert_path):
        return
    tmp_audio = dataPrep_utils.load_audio(wav_path, 32000)
    # Fix nan in tmp_audio
    tmp_audio = np.where(np.isnan(tmp_audio), 0, tmp_audio)
    tmp_max = np.abs(tmp_audio).max()
    if tmp_max > 2.2:
        print(f"{wav_name}-filtered,{tmp_max}")
        return
    tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha * 32768)
                   ) + ((1 - alpha) * 32768) * tmp_audio
    tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha * 1145.14)
                    ) + ((1 - alpha) * 1145.14) * tmp_audio
    tmp_audio = librosa.resample(
        tmp_audio32b, orig_sr=32000, target_sr=16000)  # 不是重采样问题
    tensor_wav16 = torch.from_numpy(tmp_audio)
    if is_half:
        tensor_wav16 = tensor_wav16.half().to(device)
    else:
        tensor_wav16 = tensor_wav16.to(device)

    ssl = model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(
        1, 2).cpu()  # torch.Size([1, 768, 215]), hubert model
    if np.isnan(ssl.detach().numpy()).sum() != 0:
        nan_fails.append((wav_name, wav_path))
        print(f"nan filtered:{wav_name}")
        return
    savePath = f"{wav32dir}/{wav_name}"
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    wavfile.write(
        savePath,
        32000,
        tmp_audio32.astype("int16"),
    )
    my_save(ssl, hubert_path)


if __name__ == "__main__":
    # config = {'inp_text': 'data/train.list', 'inp_wav_dir': '/home/px/repo/modelDimensions/gpt_sovits/data', 'exp_name': 'TestA1', 'opt_dir': 'logs/TestA1',
    #           'cnhubert_base_dir': 'GPT_SoVITS/pretrained_models/chinese-hubert-base', 'i_part': '0', 'all_parts': '2', '_CUDA_VISIBLE_DEVICES': '0', 'is_half': 'False'}
    # os.environ.update(config)

    inp_text = os.environ.get("inp_text")
    inp_wav_dir = os.environ.get("inp_wav_dir")
    exp_name = os.environ.get("exp_name")
    i_part = os.environ.get("i_part")
    all_parts = os.environ.get("all_parts")
    if "_CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

    opt_dir = os.environ.get("opt_dir")
    cnhubert.cnhubert_base_path = os.environ.get("cnhubert_base_dir")

    is_half = eval(os.environ.get("is_half", "True")
                   ) and torch.cuda.is_available()

    hubert_dir = f"{opt_dir}/4-cnhubert"
    wav32dir = f"{opt_dir}/5-wav32k"
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(hubert_dir, exist_ok=True)
    os.makedirs(wav32dir, exist_ok=True)

    maxx = 0.95
    alpha = 0.5
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = cnhubert.get_model()
    if is_half:
        model = model.half().to(device)
    else:
        model = model.to(device)

    nan_fails = []
    with open(inp_text, "rt", encoding="utf8") as fr:
        lines = fr.read().strip("\n").split("\n")
    for line in lines[int(i_part)::int(all_parts)]:
        try:
            wav_name, spk_name, language, text = line.split("|")
            wav_name = dataPrep_utils.clean_path(wav_name)
            if inp_wav_dir != "" and (not inp_wav_dir is None):
                wav_name = os.sep.join(wav_name.split(os.sep)[1:])
                wav_path = f"{inp_wav_dir}/{wav_name}"
            else:
                wav_path = wav_path

            name2go(wav_name, wav_path)
        except:
            print(line, traceback.format_exc())

    if len(nan_fails) > 0 and is_half:
        is_half = False
        model = model.float()

        for wav in nan_fails:
            try:
                name2go(wav[0], wav[1])
            except:
                print(wav_name, traceback.format_exc())
