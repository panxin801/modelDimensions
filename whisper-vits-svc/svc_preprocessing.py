import os
import sys
import torch
import subprocess
import argparse


assert torch.cuda.is_available(), "\033[31m You need GPU to Train! \033[0m"
print("CPU Count is :", os.cpu_count())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=int, default=0, help="thread count")
    args = parser.parse_args()

    commands = [
        f"python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000 -t {args.t}",
        f"python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-32k -s 32000 -t {args.t}",
        "python prepare/preprocess_crepe.py -w data_svc/waves-16k/ -p data_svc/pitch",
        "python prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper",
        "python prepare/preprocess_hubert.py -w data_svc/waves-16k/ -v data_svc/hubert",
        f"python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker -t {args.t}",
        "python prepare/preprocess_speaker_ave.py data_svc/speaker/ data_svc/singer",
        f"python prepare/preprocess_spec.py -w data_svc/waves-32k/ -s data_svc/specs -t {args.t}",
        "python prepare/preprocess_train.py",
        "python prepare/preprocess_zzz.py",
    ]

    for command in commands:
        print(f"Command: {command}")

        process = subprocess.Popen(command, shell=True)
        outcode = process.wait()
        if (outcode):
            break
