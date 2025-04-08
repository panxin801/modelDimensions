from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import torch
from scipy.io.wavfile import write
from omegaconf import OmegaConf

from meldataset import (mel_spectrogram, MAX_WAV_VALUE, load_wav)
from models import Generator


configs = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, configs.n_fft, configs.num_mels, configs.sampling_rate, configs.hop_size, configs.win_size, configs.fmin, configs.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "*")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ""
    return sorted(cp_list)[-1]


def inference(configs):
    generator = Generator(configs).to(device)

    state_dict_g = load_checkpoint(configs.checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])

    filelist = os.listdir(configs.input_wavs_dir)

    os.makedirs(configs.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            wav, sr = load_wav(os.path.join(configs.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0))
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")

            output_file = os.path.join(configs.output_dir, os.path.splitext(filname)[
                                       0] + "_generated.wav")
            write(output_file, configs.sampling_rate, audio)
            print(output_file)


def main():
    print("Initializing Inference Process..")
    global configs

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wavs_dir", default="test_files")
    parser.add_argument("--output_dir", default="generated_files")
    parser.add_argument("--checkpoint_file", required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(
        a.checkpoint_file)[0], "config.yaml")
    configs = OmegaConf.load(config_file)
    configs.input_wavs_dir = a.input_wavs_dir
    configs.output_dir = a.output_dir
    configs.checkpoint_file = a.checkpoint_file

    torch.manual_seed(configs.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(configs.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    inference(configs)


if __name__ == "__main__":
    main()
