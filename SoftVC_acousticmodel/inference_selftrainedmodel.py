import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from acoustic import AcousticModel


def generate(args):
    print("Loading acoustic model checkpoint")
    acoustic = AcousticModel(discrete=False, upsample=True)

    ckpt = "output/model-best.pt"
    stateDict = torch.load(ckpt, map_location="cpu")
    newstateDict = {}
    for k, v in stateDict["acoustic-model"].items():
        if k.startswith("module."):
            eles = k.split(".", 1)
            newstateDict[".".join(eles[1:])] = v
        else:
            newstateDict[k] = v
    acoustic.load_state_dict(newstateDict)
    acoustic.cuda().eval()

    print(f"Generating from {args.in_dir} -> {args.out_dir}")
    for path in tqdm(list(args.in_dir.rglob("*.npy"))):
        units = np.load(path)
        units_dtype = torch.long if args.model == "discrete" else torch.float
        units = torch.tensor(units, dtype=units_dtype).cuda()

        with torch.inference_mode():
            mel_ = acoustic.generate(units.unsqueeze(0))
            mel_ = mel_.transpose(1, 2)

        out_path = args.out_dir / path.relative_to(args.in_dir)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(out_path.with_suffix(".npy"), mel_.squeeze().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate spectrograms from input speech units (discrete or soft)."
    )
    parser.add_argument(
        "model",
        help="available models (HuBERT-Soft or HuBERT-Discrete)",
        choices=["soft", "discrete"]
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    args = parser.parse_args()
    generate(args)
