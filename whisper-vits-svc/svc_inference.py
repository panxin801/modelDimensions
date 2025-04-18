import os
import torch
import sys
import argparse
from omegaconf import OmegaConf


def main(args):
    if (args.ppg == None):
        args.ppg = "svc_tmp.ppg.npy"
        print(
            f"Auto run : python whisper/inference.py -w {args.wave} -p {args.ppg}")
        os.system(f"python whisper/inference.py -w {args.wave} -p {args.ppg}")

    if (args.vec == None):
        args.vec = "svc_tmp.vec.npy"
        print(
            f"Auto run : python hubert/inference.py -w {args.wave} -v {args.vec}")
        os.system(f"python hubert/inference.py -w {args.wave} -v {args.vec}")

    if (args.pit == None):
        args.pit = "svc_tmp.pit.csv"
        print(
            f"Auto run : python pitch/inference.py -w {args.wave} -p {args.pit}")
        os.system(f"python pitch/inference.py -w {args.wave} -p {args.pit}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = OmegaConf.load(args.config)
    print(hp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument("--model", type=str, required=True,
                        help="path of model for evaluation")
    parser.add_argument("--wave", type=str, required=True,
                        help="Path of raw audio.")
    parser.add_argument("--spk", type=str, required=True,
                        help="Path of speaker.")
    parser.add_argument("--ppg", type=str,
                        help="Path of content vector.")
    parser.add_argument("--vec", type=str,
                        help="Path of hubert vector.")
    parser.add_argument("--pit", type=str,
                        help="Path of pitch csv file.")
    parser.add_argument("--shift_l", type=int, default=0,
                        help="Pitch shift key for [shift_l, shift_r]")
    parser.add_argument("--shift_r", type=int, default=0,
                        help="Pitch shift key for [shift_l, shift_r]")
    args = parser.parse_args()

    assert args.shift_l >= -12  # shift_l should be in [-12, 12]
    assert args.shift_r >= -12  # shift_r should be in [-12, 12]
    assert args.shift_l <= 12
    assert args.shift_r <= 12
    assert args.shift_l <= args.shift_r

    os.makedirs("./svc_out", exist_ok=True)

    main(args)
