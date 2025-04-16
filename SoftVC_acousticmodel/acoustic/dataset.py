import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
from pathlib import Path


class MelDataset(data.Dataset):
    def __init__(self, root: Path, train: bool = True, discrete: bool = False):
        self.discrete = discrete
        self.mels_dir = root / "mels"
        self.units_dir = root / "discrete" if discrete else root / "soft"

        pattern = "train/**/*.npy" if train else "dev/**/*.npy"
        self.metadata = [
            path.relative_to(self.mels_dir).with_suffix("")
            for path in self.mels_dir.rglob(pattern)
        ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        """
        Args:
        index: int 

        Return:
        mel: [F_mel,D] F is frame, D=128
        units: [F_units,D] F is frame, D=256
        """
        path = self.metadata[index]
        mel_path = self.mels_dir / path
        units_path = self.units_dir / path

        mel = np.load(mel_path.with_suffix(".npy")).T
        units = np.load(units_path.with_suffix(".npy"))

        length = 2 * units.shape[0]

        mel = torch.from_numpy(mel[:length, :])
        mel = F.pad(mel, (0, 0, 1, 0))
        units = torch.from_numpy(units)
        if self.discrete:
            units = units.long()
        return mel, units

    def pad_collate(self, batch):
        """
        Args:
        batch: ...

        Return:
        mel: [B, F_mel,D] F is frame, D=128
        mels_lengths: [B] saved lengths
        units: [B, F_units,D] F is frame, D=256
        units_lengths: [B] saved lengths
        """
        mels, units = zip(*batch)

        mels, units = list(mels), list(units)

        mels_lengths = torch.tensor([x.size(0) - 1 for x in mels])
        units_lengths = torch.tensor([x.size(0) for x in units])

        mels = nn.utils.rnn.pad_sequence(mels, batch_first=True)
        units = nn.utils.rnn.pad_sequence(units, batch_first=True,
                                          padding_value=100 if self.discrete else 0)

        return mels, mels_lengths, units, units_lengths
