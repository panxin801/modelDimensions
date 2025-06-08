import librosa
import torch
import torchaudio
import json
import os
import sys
import argparse
import logging
import shutil
from time import time as ttime

logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logger = logging


def my_save(fea, path):  # fix issue: torch.save doesn't support chinese path
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s.pth" % (ttime())
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir, name))


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info("Saving model and optimizer state at iteration {} to {}".format(
        iteration, checkpoint_path))
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    # torch.save(
    my_save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def load_wav_to_torch_and_resample(file_path: str, target_sample_rate: int = 16000) -> torch.Tensor:
    # 加载音频文件
    waveform, sample_rate = librosa.load(file_path, sr=None)

    # 如果采样率与目标采样率不同，则进行重采样
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(torch.from_numpy(waveform))

    # 返回音频张量
    return waveform


def load_wav_to_torch(full_path):
    data, sampling_rate = librosa.load(full_path, sr=None)
    return torch.FloatTensor(data), sampling_rate


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def get_hparams(init=True, stage=1):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/s2.json",
        help="JSON file for configuration",
    )
    parser.add_argument("-p", "--pretrain", type=str,
                        required=False, default=None, help="pretrain dir")
    parser.add_argument(
        "-rs",
        "--resume_step",
        type=int,
        required=False,
        default=None,
        help="resume step",
    )
    # parser.add_argument('-e', '--exp_dir', type=str, required=False,default=None,help='experiment directory')
    # parser.add_argument('-g', '--pretrained_s2G', type=str, required=False,default=None,help='pretrained sovits gererator weights')
    # parser.add_argument('-d', '--pretrained_s2D', type=str, required=False,default=None,help='pretrained sovits discriminator weights')

    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.pretrain = args.pretrain
    hparams.resume_step = args.resume_step
    # hparams.data.exp_dir = args.exp_dir
    if stage == 1:
        model_dir = hparams.s1_ckpt_dir
    else:
        model_dir = hparams.s2_ckpt_dir
    config_save_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(config_save_path, "w") as f:
        f.write(data)
    return hparams


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.ERROR)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


if __name__ == "__main__":
    print(
        load_wav_to_torch(
            "/home/fish/wenetspeech/dataset_vq/Y0000022499_wHFSeHEx9CM/S00261.flac",
        )
    )
