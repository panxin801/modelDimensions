import logging
import sys
import os
import torch
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


def getLogger(modelDir, fileName="train.log"):
    global logger

    logger = logging.getLogger(os.path.basename(modelDir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    h = logging.FileHandler(os.path.join(modelDir, fileName))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)

    return logger


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(cla, mean=0.0, std=0.01):
    clsName = cla.__class__.__name__
    if clsName.find("Conv") != -1:
        cla.weight.data.normal_(mean, std)


def apply_weight_norm(cla):
    clsName = cla.__class__.__name__
    if clsName.find("Conv") != -1:
        torch.nn.utils.weight_norm(cla)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location="cpu")
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print(f"Saving checkpoint to {filepath}")
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "???????")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]
