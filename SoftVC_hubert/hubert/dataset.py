import random
import torch
import torchaudio
import torch.nn.functional as F
import torch.utils.data as data
import json
import numpy as np
from pathlib import Path


class AcousticUnitsDataset(data.Dataset):
    def __init__(self,
                 root: Path,
                 sample_rate: int = 16000,
                 label_rate: int = 50,
                 min_samples: int = 32000,
                 max_samples: int = 250000,
                 train: bool = True
                 ):
        self.wavs_dir = root / "wavs"
        self.units_dir = root / "discrete"

        with open(root / "lengths.json", "rt", encoding="utf8") as file:
            self.lengths = json.load(file)

        # I use wav files instead of flac
        # pattern = "train-*/**/*.flac" if train else "dev-*/**/*.flac"
        pattern = "train-*/**/*.wav" if train else "dev-*/**/*.wav"
        metadata = (
            (path, path.relative_to(self.wavs_dir).with_suffix("").as_posix())
            for path in self.wavs_dir.rglob(pattern)
        )
        metadata = ((path, key)
                    for path, key in metadata if key in self.lengths)
        self.metadata = [
            path for path, key in metadata if self.lengths[key] > min_samples
        ]

        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.train = train

        self.resample = torchaudio.transforms.Resample(22050, 16000)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        """
        Args:
        index: int

        Return:
        wav: [1,T] T means wav samples
        codes: [F]  F means frames. codes are integers.
        """
        wav_path = self.metadata[index]  # wav_path
        units_path = self.units_dir / wav_path.relative_to(self.wavs_dir)

        # wav.size()=[1,T] T means wav samples
        wav, _ = torchaudio.load(wav_path)
        # Self add wav resample
        wav = self.resample(wav)
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        # codes are discrete units from hubert
        codes = np.load(units_path.with_suffix(
            ".npy"))  # codes.shape=[F]  F means frames. codes are integers.

        return wav, torch.from_numpy(codes).long()

    def collate(self, batch):
        """
        Args:
        batch are the following: 
        wav: [1,T] T means wav samples
        codes: [F]  F means frames. codes are integers.

        Return:
        wavs: [B,1,T] T means wav samples
        codes: [B,F] F means frames
        """
        wavs, codes = zip(*batch)
        wavs, codes = list(wavs), list(codes)

        wav_lengths = [wav.size(-1) for wav in wavs]
        code_lengths = [code.size(-1) for code in codes]

        wav_frames = min(self.max_samples, *wav_lengths)

        collated_wavs, wav_offsets = [], []
        for wav in wavs:
            wav_diff = wav.size(-1) - wav_frames
            wav_offset = random.randint(0, wav_diff)
            wav = wav[:, wav_offset: wav_offset + wav_frames]

            collated_wavs.append(wav)
            wav_offsets.append(wav_offset)

        rate = self.label_rate / self.sample_rate
        code_offsets = [round(wav_offset * rate) for wav_offset in wav_offsets]
        code_frames = round(wav_frames * rate)
        remaining_code_frames = [
            length - offset for length, offset in zip(code_lengths, code_offsets)
        ]
        code_frames = min(code_frames, *remaining_code_frames)

        collated_codes = []
        for code, code_offset in zip(codes, code_offsets):
            code = code[code_offset: code_offset + code_frames]
            collated_codes.append(code)

        wavs = torch.stack(collated_wavs, dim=0)
        codes = torch.stack(collated_codes, dim=0)

        return wavs, codes
