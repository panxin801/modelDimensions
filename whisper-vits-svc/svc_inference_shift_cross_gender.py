import sys
import os
import torch
import argparse
import numpy as np
from omegaconf import OmegaConf
from scipy.io.wavfile import write
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from pitch import load_csv_pitch
from vits.models import SynthesizerInfer
from svc_inference import load_svc_model, svc_infer
from feature_retrieval import (
    IRetrieval, DummyRetrieval, FaissIndexRetrieval, load_retrieve_index)


def f0_static(path_pitchs):
    """ Get f0 static of target spk
    """
    if os.path.isdir(path_pitchs):  # Dir
        f0s = []
        for roots, dirs, files in os.walk(path_pitchs):
            for file in files:
                if file.endswith(".npy"):  # ".pit.npy" or ".npy" files
                    file_f0 = os.path.join(roots, file)
                    f0 = np.load(file_f0)
                    nonzero_indices = np.nonzero(f0)

                    if len(f0s) == 0:
                        f0s = np.log(f0[nonzero_indices])
                    else:
                        f0s = np.r_[f0s, np.log(f0[nonzero_indices])]
    elif os.path.isfile(path_pitchs):  # file
        f0 = np.load(path_pitchs)
        nonzero_indices = np.nonzero(f0)
        f0s = np.log(f0[nonzero_indices])

    f0stats = np.array([np.mean(f0s), np.std(f0s)])
    return f0stats


def f0_shift(pit, f0stats):
    f0 = np.array(pit)
    nonzero_indices = np.nonzero(f0)

    f0_nonezero = np.log(f0[nonzero_indices])
    src_f0_stats = np.array([np.mean(f0_nonezero), np.std(f0_nonezero)])
    tgt_f0_stats = f0stats

    T = len(f0)
    # perform f0 conversion
    cvf0 = np.zeros(T)
    nonzero_indices = f0 > 0
    cvf0[nonzero_indices] = np.exp(
        (tgt_f0_stats[1] / src_f0_stats[1]) *
        (np.log(f0[nonzero_indices]) - src_f0_stats[0]) + tgt_f0_stats[0])
    return cvf0


def get_speaker_name_from_path(speaker_path: Path) -> str:
    suffixes = "".join(speaker_path.suffixes)
    filename = speaker_path.stem
    return filename.rstrip(suffixes)


def create_retrival(cli_args) -> IRetrieval:
    if not cli_args.enable_retrieval:
        print("infer without retrival")
        return DummyRetrieval()
    else:
        print("load index retrival model")

    speaker_name = get_speaker_name_from_path(Path(cli_args.spk))
    base_path = Path(".").absolute() / "data_svc" / "indexes" / speaker_name

    if cli_args.hubert_index_path:
        hubert_index_filepath = cli_args.hubert_index_path
    else:
        index_name = f"{cli_args.retrieval_index_prefix}hubert.index"
        hubert_index_filepath = base_path / index_name

    if cli_args.whisper_index_path:
        whisper_index_filepath = cli_args.whisper_index_path
    else:
        index_name = f"{cli_args.retrieval_index_prefix}whisper.index"
        whisper_index_filepath = base_path / index_name

    return FaissIndexRetrieval(
        hubert_index=load_retrieve_index(
            filepath=hubert_index_filepath,
            ratio=cli_args.retrieval_ratio,
            n_nearest_vectors=cli_args.n_retrieval_vectors
        ),
        whisper_index=load_retrieve_index(
            filepath=whisper_index_filepath,
            ratio=cli_args.retrieval_ratio,
            n_nearest_vectors=cli_args.n_retrieval_vectors
        ),
    )


def main(args):
    if (args.ppg == None):
        args.ppg = "svc_tmp.ppg.npy"
        print(
            f"Auto run : python whisper/inference.py -w {args.wave} -p {args.ppg}")
        # os.system(f"python whisper/inference.py -w {args.wave} -p {args.ppg}")

    if (args.vec == None):
        args.vec = "svc_tmp.vec.npy"
        print(
            f"Auto run : python hubert/inference.py -w {args.wave} -v {args.vec}")
        # os.system(f"python hubert/inference.py -w {args.wave} -v {args.vec}")

    if (args.pit == None):
        args.pit = "svc_tmp.pit.csv"
        print(
            f"Auto run : python pitch/inference.py -w {args.wave} -p {args.pit}")
        # os.system(f"python pitch/inference.py -w {args.wave} -p {args.pit}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = OmegaConf.load(args.config)
    model = SynthesizerInfer(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp)
    load_svc_model(args.model, model)
    model.eval()
    model.to(device)

    spk = np.load(args.spk)
    spk = torch.FloatTensor(spk)

    # Dir of target speaker's f0 files， assume train spk 001
    f0stats = f0_static("data_svc/pitch/001")

    ppg = np.load(args.ppg)
    ppg = np.repeat(ppg, 2, 0)
    ppg = torch.FloatTensor(ppg)

    vec = np.load(args.vec)
    vec = np.repeat(vec, 2, 0)
    vec = torch.FloatTensor(vec)

    shift = args.shift
    print(f"pitch shift: {shift}")

    pit = load_csv_pitch(args.pit)
    pit = f0_shift(pit, f0stats)
    pit = torch.FloatTensor(pit)

    retrieval = create_retrival(args)

    out_audio = svc_infer(model, retrieval, spk, pit, ppg, vec, hp, device)
    write(os.path.join("./_svc_out",
          f"svc_out_{shift}.wav"), hp.data.sampling_rate, out_audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('--model', type=str, required=True,
                        help="path of model for evaluation")
    parser.add_argument('--wave', type=str, required=True,
                        help="Path of raw audio.")
    parser.add_argument('--spk', type=str, required=True,
                        help="Path of speaker.")
    parser.add_argument('--ppg', type=str,
                        help="Path of content vector.")
    parser.add_argument('--vec', type=str,
                        help="Path of hubert vector.")
    parser.add_argument('--pit', type=str,
                        help="Path of pitch csv file.")
    parser.add_argument("--shift", action="store_true", default=0,
                        help="Pitch shift value using for cross gender.")
    parser.add_argument('--enable-retrieval', action="store_true",
                        help="Enable index feature retrieval")
    args = parser.parse_args()

    assert args.shift >= 0 and args.shift <= 12, "Pitch shift value should be between 0 and 12."

    os.makedirs("./_svc_out", exist_ok=True)

    main(args)
