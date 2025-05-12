import json
import os
import glob
import shutil
import logging
import torch
import subprocess
import numpy as np
from scipy.io.wavfile import read

MATPLOTLIB_FLAG = False

logger = logging.getLogger(__name__)


def download_checkpoint(
    dir_path, repo_config, token=None, regex="G_*.pth", mirror="openi"
):
    repo_id = repo_config["repo_id"]
    f_list = glob.glob(os.path.join(dir_path, regex))
    if f_list:
        print("Use existed model, skip downloading.")
        return
    if mirror.lower() == "openi":
        import openi

        kwargs = {"token": token} if token else {}
        openi.login(**kwargs)

        model_image = repo_config["model_image"]
        openi.model.download_model(repo_id, model_image, dir_path)

        fs = glob.glob(os.path.join(dir_path, model_image, "*.pth"))
        for file in fs:
            shutil.move(file, dir_path)
        shutil.rmtree(os.path.join(dir_path, model_image))
    else:
        for file in ["DUR_0.pth", "D_0.pth", "G_0.pth"]:
            hf_hub_download(
                repo_id, file, local_dir=dir_path, local_dir_use_symlinks=False
            )


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams_from_file(config_path):
    # print("config_path: ", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warning(
            "{} is not a git repository, therefore hash value comparison will be ignored.".format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warning(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
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
