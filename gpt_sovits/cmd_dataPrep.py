import sys
import os
import argparse
import torch
from subprocess import Popen

from tools import dataPrep_utils
from config import (
    exp_root,
    infer_device,
    is_half,
    is_share,
    python_exec,
)


# 判断是否有能用来训练和加速推理的N卡
ok_gpu_keywords = {
    "10",
    "16",
    "20",
    "30",
    "40",
    "A2",
    "A3",
    "A4",
    "P4",
    "A50",
    "500",
    "A60",
    "70",
    "80",
    "90",
    "M4",
    "T4",
    "TITAN",
    "L4",
    "4060",
    "H",
    "600",
    "506",
    "507",
    "508",
    "509",
}
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False
set_gpu_numbers = set()

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper() for value in ok_gpu_keywords):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            set_gpu_numbers.add(i)
            mem.append(int(torch.cuda.get_device_properties(
                i).total_memory / 1024 / 1024 / 1024 + 0.4))
# # 判断是否支持mps加速
# if torch.backends.mps.is_available():
#     if_gpu_ok = True
#     gpu_infos.append("%s\t%s" % ("0", "Apple GPU"))
#     mem.append(psutil.virtual_memory().total/ 1024 / 1024 / 1024) # 实测使用系统内存作为显存不会爆显存

gpus = "-".join([i[0] for i in gpu_infos])
default_gpu_numbers = str(sorted(list(set_gpu_numbers))[0])


def fix_gpu_number(input):  # 将越界的number强制改到界内
    try:
        if int(input) not in set_gpu_numbers:
            return default_gpu_numbers
    except:
        return input
    return input


def run1a(dataLst, expname, gpuNumbers, bertPretrainDir):
    ps1a = []  # 子线程列表
    dataLst = dataPrep_utils.clean_path(dataLst)
    wavDir = dataPrep_utils.clean_path(
        os.path.abspath(os.path.dirname(dataLst)))
    if dataPrep_utils.check_for_existance([dataLst, wavDir], is_dataset_processing=True):
        dataPrep_utils.check_details(
            [dataLst, wavDir], is_dataset_processing=True)

    if ps1a == []:
        opt_dir = os.sep.join([exp_root, expname])
        config = {
            "inp_text": dataLst,
            "inp_wav_dir": wavDir,
            "exp_name": expname,
            "opt_dir": opt_dir,
            "bert_pretrained_dir": bertPretrainDir,
        }
        gpuNames = gpuNumbers.split("-")
        allparts = len(gpuNames)
        for i in range(allparts):
            config.update(
                {
                    "i_part": str(i),
                    "all_parts": str(allparts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpuNames[i]),
                    "is_half": str(is_half),
                }
            )
            print(config)
            os.environ.update(config)
            cmd = f"{python_exec} GPT_SoVITS/prepare_datasets/1-get-text.py"
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1a.append(p)

        for p in ps1a:
            p.wait()

        opt = []
        for i in range(allparts):
            txtPath = f"{opt_dir}/2-name2text-{i}.txt"
            with open(txtPath, "rt", encoding="utf8") as fr:
                opt += fr.read().strip("\n").split("\n")
            os.remove(txtPath)

        pathText = f"{opt_dir}/2-name2text.txt"
        with open(pathText, "wt", encoding="utf8") as fw:
            fw.write("\n".join(opt) + "\n")

    print(f"Finish 1a:{pathText}")


def run1b(dataLst, expname, gpuNumbers, hubertPretrainDir):
    ps1b = []
    dataLst = dataPrep_utils.clean_path(dataLst)
    wavDir = dataPrep_utils.clean_path(
        os.path.abspath(os.path.dirname(dataLst)))
    if dataPrep_utils.check_for_existance([dataLst, wavDir], is_dataset_processing=True):
        dataPrep_utils.check_details(
            [dataLst, wavDir], is_dataset_processing=True)

    if ps1b == []:
        opt_dir = os.sep.join([exp_root, expname])
        config = {
            "inp_text": dataLst,
            "inp_wav_dir": wavDir,
            "exp_name": expname,
            "opt_dir": opt_dir,
            "cnhubert_base_dir": hubertPretrainDir,
        }
        gpuNames = gpuNumbers.split("-")
        allparts = len(gpuNames)
        for i in range(allparts):
            config.update(
                {
                    "i_part": str(i),
                    "all_parts": str(allparts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpuNames[i]),
                    "is_half": str(is_half),
                }
            )
            print(config)
            os.environ.update(config)
            cmd = f"{python_exec} GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1b.append(p)

        for p in ps1b:
            p.wait()

    print("Finish 1b")


def main(args):
    """ Stage document.
    0 -> 文本分词与特征提取
    1 -> 语音自监督特征提取
    2 -> 语义Token提取
    """

    if args.start <= 0 and args.end >= 0:
        print("Stage 0: 文本分词与特征提取")

        gpuNumbers = "0-0"  # GPU卡号以-分割，每个卡号一个进程
        bertPretrainDir = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

        run1a(args.dataLst, args.expname, gpuNumbers, bertPretrainDir)

    if args.start <= 1 and args.end >= 1:
        print("Stage 1: 语音自监督特征提取")

        gpuNumbers = "0-0"
        hubertPretrainDir = "GPT_SoVITS/pretrained_models/chinese-hubert-base"

        run1b(args.dataLst, args.expname, gpuNumbers, hubertPretrainDir)

    if args.start <= 2 and args.end >= 2:
        print("Stage 2: 语义Token提取")

    print("All ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # version 只能是v1 v2 v3 v4
    parser.add_argument("version", type=str, default="v4",
                        choices=["v1", "v2", "v3", "v4"], help="version of gpt_sovits to use")
    parser.add_argument("--start", "-s", type=int, default=0,
                        help="start index of data to preprocess")
    parser.add_argument("--end", "-e", type=int, default=10,
                        help="end index of data to preprocess")
    parser.add_argument("--dataLst", type=str, default="data/train.list",
                        help="path to data list file")
    parser.add_argument("--expname", type=str,
                        default="TestA1", help="name of experiment")
    args = parser.parse_args()

    main(args)
