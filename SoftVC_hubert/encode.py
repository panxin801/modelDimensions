# Inference code for dataset using hubert-{discrete, soft}

import argparse
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from pathlib import Path


def encode_dataset(args):
    print(f"Loading hubert checkpoint")
    hubert = torch.hub.load("bshall/hubert:main",
                            f"hubert_{args.model}",
                            trust_repo=True).cuda()

    print(f"Encoding dataset at {args.in_dir}")
    for in_path in tqdm(list(args.in_dir.rglob(f"*{args.extension}"))):
        wav, sr = torchaudio.load(in_path)
        wav = torchaudio.functional.resample(
            wav, sr, 16000)  # resampling to 16kHz
        wav = wav.unsqueeze(0).cuda()

        # inference_mode 在1.9新加入，和no_grad相比，都不计算梯度，而且inference_mode做了强制关闭梯度记录。并且不能在中途设置梯度。
        with torch.inference_mode():
            units = hubert.units(wav)

        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), units.squeeze().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument("model",
                        help="available models (HuBERT-Soft or HuBERT-Discrete)",
                        choices=["soft", "discrete"])
    parser.add_argument("in_dir",
                        metavar="in-dir",
                        help="path to the dataset directory.",
                        type=Path)  # argparse 模块会在解析命令行参数时自动将该参数转换为 pathlib.Path 对象。
    parser.add_argument("out_dir",
                        metavar="out-dir",
                        help="path to the output directory.",
                        type=Path)
    parser.add_argument("--extension",
                        help="extension of the audio files (defaults to .flac).",
                        default=".flac",
                        type=str)
    args = parser.parse_args()

    encode_dataset(args)
