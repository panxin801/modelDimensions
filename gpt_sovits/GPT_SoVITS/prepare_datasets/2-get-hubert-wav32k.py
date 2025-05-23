import sys
import os
import torch
import traceback
import numpy as np
import librosa
from scipy.io import wavfile

# Add path
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from feature_extractor import cnhubert
from tools import dataPrep_utils

if __name__ == "__main__":
    inp_text = os.environ.get("inp_text")
    inp_wav_dir = os.environ.get("inp_wav_dir")
    exp_name = os.environ.get("exp_name")
    i_part = os.environ.get("i_part")
    all_parts = os.environ.get("all_parts")
    if "_CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

    opt_dir = os.environ.get("opt_dir")
    cnhubert.cnhubert_base_path = os.environ.get("cnhubert_base_dir")

    is_half = eval(os.environ.get("is_half", "True")
                   ) and torch.cuda.is_available()
