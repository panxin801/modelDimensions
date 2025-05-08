# cmd preprocess Bert_vits2 data.

import os
import sys
import shutil
import json
import subprocess


def get_path(data_dir):
    start_path = os.path.join("./data", data_dir)
    lbl_path = os.path.join(start_path, "esd.list")
    train_path = os.path.join(start_path, "train.list")
    val_path = os.path.join(start_path, "val.list")
    config_path = os.path.join(start_path, "configs", "config.json")

    return start_path, lbl_path, train_path, val_path, config_path


def generate_config(data_dir, batch_size):
    assert data_dir != "", "数据集名称不能为空"

    start_path, _, train_path, val_path, config_path = get_path(data_dir)
    if os.path.isfile(config_path):
        config = json.load(open(config_path, "rt", encoding="utf8"))
    else:
        config = json.load(open("configs/config.json", "rt", encoding="utf8"))

    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["train"]["batch_size"] = batch_size
    out_path = os.path.join(start_path, "configs")
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    model_path = os.path.join(start_path, "models")
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    with open(config_path, "wt", encoding="utf8") as fw:
        json.dump(config, fw, ensure_ascii=False, indent=4)

    if not os.path.exists("config.yml"):
        shutil.copy("default_config.yml", "config.yml")

    return True  # 配置生成成功


def resample(data_dir):
    assert data_dir != "", "数据集名称不能为空"

    # start_path, _, _, _, config_path = get_path(data_dir)
    # in_dir = os.path.join(start_path, "raw")
    # out_dir = os.path.join(start_path, "wavs")
    # subprocess.run(["python",
    #                 "resample_legacy.py",
    #                 "--sr",
    #                 "44100",
    #                 "--in_dir",
    #                 f"{in_dir}",
    #                 "--out_dir",
    #                 f"{out_dir}",]
    #                )
    return True


def preprocess_text(data_dir):
    assert data_dir != "", "数据集名称不能为空"

    start_path, lbl_path, train_path, val_path, config_path = get_path(
        data_dir)
    lines = open(lbl_path, "rt", encoding="utf8").readlines()
    with open(lbl_path, "wt", encoding="utf8") as fw:
        for line in lines:
            path, spk, language, text = line.strip().split("|", 3)
            path = os.path.join(start_path, "wavs",
                                os.path.basename(path)).replace("\\", "/")
            fw.writelines(f"{path}|{spk}|{language}|{text}\n")

    subprocess.run(["python",
                    "preprocess_text.py",
                    "--transcription-path",
                    f"{lbl_path}",
                    "--train-path",
                    f"{train_path}",
                    "--val-path",
                    f"{val_path}",
                    "--config-path",
                    f"{config_path}",]
                   )
    return True


if __name__ == "__main__":
    '''
    "# Bert-VITS2 数据预处理\n"
    "## 预先准备：\n"
    "下载 BERT 和 WavLM 模型：\n"
    "- [中文 RoBERTa](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)\n"
    "- [日文 DeBERTa](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm)\n"
    "- [英文 DeBERTa](https://huggingface.co/microsoft/deberta-v3-large)\n"
    "- [WavLM](https://huggingface.co/microsoft/wavlm-base-plus)\n"
    "\n"
    "将 BERT 模型放置到 `bert` 文件夹下，WavLM 模型放置到 `slm` 文件夹下，覆盖同名文件夹。\n"
    "\n"
    "数据准备：\n"
    "将数据放置在 data 文件夹下，按照如下结构组织：\n"
    "\n"
    "```\n"
    "├── data\n"
    "│   ├── {你的数据集名称}\n"
    "│   │   ├── esd.list\n"
    "│   │   ├── raw\n"
    "│   │   │   ├── ****.wav\n"
    "│   │   │   ├── ****.wav\n"
    "│   │   │   ├── ...\n"
    "```\n"
    "\n"
    "其中，`raw` 文件夹下保存所有的音频文件，`esd.list` 文件为标签文本，格式为\n"
    "\n"
    "```\n"
    "****.wav|{说话人名}|{语言 ID}|{标签文本}\n"
    "```\n"
    "\n"
    "例如：\n"
    "```\n"
    "vo_ABDLQ001_1_paimon_02.wav|派蒙|ZH|没什么没什么，只是平时他总是站在这里，有点奇怪而已。\n"
    "noa_501_0001.wav|NOA|JP|そうだね、油断しないのはとても大事なことだと思う\n"
    "Albedo_vo_ABDLQ002_4_albedo_01.wav|Albedo|EN|Who are you? Why did you alarm them?\n"
    "...\n"
    "```\n"
    '''

    data_dir = "LJ0508"
    # 第一步：生成配置文件
    # 批大小（Batch size）：24 GB 显存可用 12
    batch_size = 3
    result = generate_config(data_dir, batch_size)
    if not result:
        sys.exit(1)

    # 第二步：预处理音频文件
    result = resample(data_dir)

    # 第三步：预处理标签文件
    result = preprocess_text(data_dir)
