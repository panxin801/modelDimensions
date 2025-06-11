import argparse
import os
import sys
import json
import yaml
import shutil
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


pretrained_sovits_name = [
    "GPT_SoVITS/pretrained_models/s2G488k.pth",
    "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    "GPT_SoVITS/pretrained_models/s2Gv3.pth",
    "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
]
pretrained_gpt_name = [
    "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "GPT_SoVITS/pretrained_models/s1v3.ckpt",
]
SoVITS_weight_root = ["SoVITS_weights", "SoVITS_weights_v2",
                      "SoVITS_weights_v3", "SoVITS_weights_v4"]
GPT_weight_root = ["GPT_weights", "GPT_weights_v2",
                   "GPT_weights_v3", "GPT_weights_v4"]

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
set_gpu_numbers = set()
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper() for value in ok_gpu_keywords):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            set_gpu_numbers.add(i)
default_gpu_numbers = str(sorted(list(set_gpu_numbers))[0])


def clean_dir(tmp):
    if os.path.exists(tmp):
        for name in os.listdir(tmp):
            if name == "jieba.cache":
                continue
            path = os.path.join(tmp, name)
            delete = os.remove if os.path.isfile(path) else shutil.rmtree
            try:
                delete(path)
            except Exception as e:
                print(str(e))


def fix_gpu_number(input):  # 将越界的number强制改到界内
    try:
        if int(input) not in set_gpu_numbers:
            return default_gpu_numbers
    except:
        return input
    return input


def fix_gpu_numbers(inputs):
    output = []
    try:
        for input in inputs.split(","):
            output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs


def finetunesovits(batch_size,
                   total_epoch,
                   exp_name,
                   text_low_lr_rate,
                   if_save_latest,
                   if_save_every_weights,
                   save_every_epoch,
                   gpu_numbers1Ba,
                   pretrained_s2G,
                   pretrained_s2D,
                   if_grad_ckpt,
                   lora_rank,
                   version):
    """ Finetune sovits
    """

    with open("GPT_SoVITS/configs/s2.json", "rt", encoding="utf8") as fr:
        data = fr.read()
        data = json.loads(data)

    s2_dir = f"{exp_root}/{exp_name}"
    os.makedirs(f"{s2_dir}/logs_s2_{version}", exist_ok=True)

    if dataPrep_utils.check_for_existance([s2_dir], is_train=True):
        dataPrep_utils.check_details([s2_dir], is_train=True)
    if is_half == False:
        data["train"]["fp16_run"] = False
        batch_size = max(1, batch_size // 2)
    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["train"]["text_low_lr_rate"] = text_low_lr_rate
    data["train"]["pretrained_s2G"] = pretrained_s2G
    data["train"]["pretrained_s2D"] = pretrained_s2D
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["save_every_epoch"] = save_every_epoch
    data["train"]["gpu_numbers"] = gpu_numbers1Ba
    data["train"]["grad_ckpt"] = if_grad_ckpt
    data["train"]["lora_rank"] = lora_rank
    data["model"]["version"] = version
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
    data["save_weight_dir"] = SoVITS_weight_root[int(version[-1]) - 1]
    data["name"] = exp_name
    data["version"] = version

    tmp = os.path.join(os.getcwd(), "TEMP")
    os.environ["TEMP"] = tmp
    os.makedirs(tmp, exist_ok=True)
    clean_dir(tmp)

    tmp_config_path = f"{tmp}/tmp_s2.json"
    with open(tmp_config_path, "wt", encoding="utf8") as f:
        f.write(json.dumps(data))

    if version in ["v1", "v2"]:
        cmd = f"{python_exec} GPT_SoVITS/s2_train.py --config {tmp_config_path}"
    else:
        cmd = f"{python_exec} GPT_SoVITS/s2_train_v3_lora.py --config {tmp_config_path}"
    print(cmd)
    p = Popen(cmd, shell=True)
    p.wait()

    print("Finetune sovits done")


def finetuneGPT(batch_size,
                total_epoch,
                exp_name,
                if_dpo,
                if_save_latest,
                if_save_every_weights,
                save_every_epoch,
                gpu_numbers1Bb,
                pretrained_s1,
                version):
    """ Finetune GPT
    """

    with open("GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml") as f:
        data = f.read()
        data = yaml.load(data, Loader=yaml.FullLoader)

    s1_dir = f"{exp_root}/{exp_name}"
    os.makedirs(f"{s1_dir}/logs_s1", exist_ok=True)
    if dataPrep_utils.check_for_existance([s1_dir], is_train=True):
        dataPrep_utils.check_details([s1_dir], is_train=True)
    if is_half is False:
        data["train"]["precision"] = "32"
        batch_size = max(1, batch_size // 2)
    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["pretrained_s1"] = pretrained_s1
    data["train"]["save_every_n_epoch"] = save_every_epoch
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_dpo"] = if_dpo
    data["train"]["half_weights_save_dir"] = GPT_weight_root[int(
        version[-1]) - 1]
    data["train"]["exp_name"] = exp_name
    data["train_semantic_path"] = f"{s1_dir}/6-name2semantic.tsv"
    data["train_phoneme_path"] = f"{s1_dir}/2-name2text.txt"
    data["output_dir"] = f"{s1_dir}/logs_s1_{version}"
    data["version"] = version

    os.environ["_CUDA_VISIBLE_DEVICES"] = fix_gpu_numbers(
        gpu_numbers1Bb.replace("-", ","))
    os.environ["hz"] = "25hz"
    tmp = os.path.join(os.getcwd(), "TEMP")
    tmp_config_path = f"{tmp}/tmp_s1.yaml"
    with open(tmp_config_path, "wt", encoding="utf8") as f:
        f.write(yaml.dump(data, default_flow_style=False))
    # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
    cmd = f"{python_exec} GPT_SoVITS/s1_train.py --config {tmp_config_path}"
    print(cmd)
    p = Popen(cmd, shell=True)
    p.wait()

    print("Finetune GPTS done")


def main(args):
    """ Stage document.
    0 -> finetune sovits
    1 -> finetune gpt
    """

    if args.start <= 0 and args.end >= 0:
        print("Stage 0: 微调Sovits")

        batch_size = 2
        total_epoch = 2
        text_low_lr_rate = 0.4  # 文本模块学习率权重[0.2, 0.6]
        if_save_latest = False  # 是否只保存最新权重
        if_save_every_weights = True  # 是否在每次保存时间点将最终小模型保存至weights文件夹
        save_every_epoch = 1
        gpu_numbers1Ba = "0"  # GPU卡号以-分割，每个卡号一个进程
        pretrained_s2G = pretrained_sovits_name[int(args.version[-1]) - 1]
        pretrained_s2D = pretrained_sovits_name[int(
            args.version[-1]) - 1].replace("s2G", "s2D")
        if_grad_ckpt = False  # v3是否开启梯度检查点节省显存占用
        lora_rank = 32  # LoRA秩, "16", "32", "64", "128"

        # finetunesovits(batch_size,
        #                total_epoch,
        #                args.expname,
        #                text_low_lr_rate,
        #                if_save_latest,
        #                if_save_every_weights,
        #                save_every_epoch,
        #                gpu_numbers1Ba,
        #                pretrained_s2G,
        #                pretrained_s2D,
        #                if_grad_ckpt,
        #                lora_rank,
        #                args.version)

    if args.start <= 1 and args.end >= 1:
        print("Stage 1: 微调GPT")

        batch_size = 2
        if_dpo = False  # 是否开启DPO训练选项(实验性)
        total_epoch = 2
        if_save_latest = False  # 是否只保存最新权重
        if_save_every_weights = True  # 是否在每次保存时间点将最终小模型保存至weights文件夹
        save_every_epoch = 1
        gpu_numbers1Bb = "0"  # GPU卡号以-分割，每个卡号一个进程
        pretrained_s1 = pretrained_gpt_name[int(args.version[-1]) - 1]

        finetuneGPT(batch_size,
                    total_epoch,
                    args.expname,
                    if_dpo,
                    if_save_latest,
                    if_save_every_weights,
                    save_every_epoch,
                    gpu_numbers1Bb,
                    pretrained_s1,
                    args.version)

    print("All ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", type=str, default="v4",
                        choices=["v1", "v2", "v3", "v4"], help="version of gpt_sovits to use")
    parser.add_argument("--start", "-s", type=int, default=0,
                        help="start index of data to preprocess")
    parser.add_argument("--end", "-e", type=int, default=10,
                        help="end index of data to preprocess")
    parser.add_argument("--expname", type=str,
                        default="TestA1", help="name of experiment")
    args = parser.parse_args()

    main(args)
