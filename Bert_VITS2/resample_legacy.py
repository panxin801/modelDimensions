import os
import argparse
from multiprocessing import (cpu_count)
from concurrent.futures import (ProcessPoolExecutor, as_completed)
import librosa
from tqdm import tqdm
from soundfile import write

from config import config


def process(item):
    wav_name, args = item
    wav_path = os.path.join(args.in_dir, wav_name)
    if os.path.exists(wav_path) and wav_path.lower().endswith(".wav"):
        wav, sr = librosa.load(wav_path, sr=args.sr)
        write(os.path.join(args.out_dir, wav_name), wav, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sr",
        type=int,
        default=config.resample_config.sampling_rate,
        help="sampling rate")
    parser.add_argument(
        "--in_dir",
        type=str,
        default=config.resample_config.in_dir,
        help="path to source dir")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=config.resample_config.out_dir,
        help="path to output dir")
    parser.add_argument(
        "--processes",
        type=int,
        default=0,
        help="cpu_processes")
    args, _ = parser.parse_known_args()
    # autodl 无卡模式会识别出46个cpu
    if args.processes == 0:
        processes = cpu_count() - 2 if cpu_count() > 4 else 1
    else:
        processes = args.processeses

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    tasks = []
    for dirpath, _, filenames in os.walk(args.in_dir):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                tasks.append((filename, args))

    with ProcessPoolExecutor(max_workers=processes) as executor:
        futures = [executor.submit(process, task) for task in tasks]

        for _ in tqdm(as_completed(futures), total=len(futures), desc="音频重采样进度"):
            pass

    print("音频重采样完成")
