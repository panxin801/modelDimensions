import torch
import torch.nn.functional as F
import random
import sys
import os
import kaldiio
import numpy as np
from scipy.io.wavfile import read
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "stage1"))

from phone import symbols
from vits.mel_processing import spectrogram_torch


class TokenVocoderDataset(torch.utils.data.Dataset):
    def __init__(self,
                 hparams,
                 config,
                 mode,):
        super().__init__()

        self.meta_file = hparams.meta_file
        self.token_scp = hparams.token_scp
        self.mel_scp = hparams.mel_scp
        self.max_wav_value = config.max_wav_value
        self.sampling_rate = config.sampling_rate
        self.filter_length = config.filter_length
        self.hop_length = config.hop_length
        self.win_length = config.win_length

        self.tokenDict = kaldiio.load_scp(self.token_scp)
        self.melDict = kaldiio.load_scp(self.mel_scp)
        self.mode = mode
        self.segmentSize = config.segment_size
        self.lengths = []
        self.specDir = "vctkDataset/totalmel_stage2"
        self.metaList = self.parse_meta_file(self.meta_file)

        os.makedirs(self.specDir, exist_ok=True)

    def parse_meta_file(self, meta_file):
        meta_list = []
        self.phone_set = symbols
        _phone_to_id = {s: i for i, s in enumerate(self.phone_set)}
        dir_name = ['p261', 'p225', 'p294', 'p347', 'p238',
                    'p234', 'p248', 'p335', 'p245', 'p326', 'p302']

        with open(meta_file, "rt", encoding="utf8") as f_meta:
            meta_lines = f_meta.read().splitlines()

        for line in tqdm(meta_lines):
            temp_list = []

            uttid, feat = line.split("||")
            if (uttid[0:4] in dir_name and self.mode == 'valid') or (uttid[0:4] not in dir_name and self.mode == 'train'):
                feat_list = feat.strip().split(" ")

                # get frame size from wav
                audiopath = os.path.join(
                    "vctkDataset/totalwav", uttid + '.wav')
                self.lengths.append(os.path.getsize(
                    audiopath) // (2 * self.hop_length))

                temp_list.append(uttid)
                temp_list.append([_phone_to_id[phone]
                                 for phone in feat_list])
                meta_list.append(temp_list)
        return meta_list

    def __len__(self,):
        return len(self.metaList)

    def __getitem__(self, index):
        uttr_id, phone = self.metaList[index]
        # token_seq = torch.from_numpy(self.tokenDict[uttr_id]) + 1
        token_seq = torch.from_numpy(self.tokenDict[uttr_id])

        mels = torch.from_numpy(self.melDict[uttr_id])
        mels_start = random.randint(0, mels.size(0))

        segment_mels = mels[mels_start: mels_start + self.segmentSize, :]

        if segment_mels.size(0) < self.segmentSize:
            segment_mels = segment_mels.permute([1, 0])
            segment_mels = F.pad(
                segment_mels, (0, self.segmentSize - segment_mels.size(1)), 'constant')
            assert segment_mels.size(0) == 80
            assert segment_mels.size(1) == self.segmentSize
        else:
            segment_mels = segment_mels.permute([1, 0])

        sampling_rate, data = read(os.path.join(
            "vctkDataset/totalwav", f"{uttr_id}.wav"))
        data = torch.FloatTensor(data.astype(np.float32))
        data_norm = data / self.max_wav_value
        data_norm = data_norm.unsqueeze(0)
        spec_filename = os.path.join(self.specDir, f"{uttr_id}.spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(
                data_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length, center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        # [L_token], [80,segmentSize], [L_wav], [filter_length/2, frame]
        return token_seq, segment_mels, data_norm.squeeze(0), spec


class TokenVocoderCollate():
    def __init__(self, segmentSize=187):
        super().__init__()

        self.segmentSize = segmentSize

    def __call__(self, batch):
        input_lengths, idx_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0,
            descending=True
        )

        max_input_len = input_lengths[0]

        token_padded = torch.LongTensor(len(batch), max_input_len).zero_()

        mels = torch.FloatTensor(len(batch), 80, self.segmentSize)

        max_spec_len = max([x[3].size(1) for x in batch])

        spec = torch.FloatTensor(len(batch))

        max_audio_len = max(x[2].size(0) for x in batch)

        audio_padded = torch.FloatTensor(len(batch), max_audio_len).zero_()

        spec_padded = torch.FloatTensor(len(batch), 101, max_spec_len).zero_()

        token_lens = torch.LongTensor(len(batch))
        spec_lens = torch.LongTensor(len(batch))
        audio_lens = torch.LongTensor(len(batch))

        for i in range(len(idx_sorted_decreasing)):
            token = batch[idx_sorted_decreasing[i]][0]
            token_padded[i, :token.size(0)] = token
            token_lens[i] = int(token.size(0) * 2)

            mel = batch[idx_sorted_decreasing[i]][1]
            mels[i, :, :] = mel

            audio = batch[idx_sorted_decreasing[i]][2]
            audio_padded[i, :audio.size(0)] = audio
            audio_lens[i] = audio.size(0)

            spec = batch[idx_sorted_decreasing[i]][3]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lens[i] = spec.size(1)

        return (token_padded, token_lens, mels, audio_padded, audio_lens, spec_padded, spec_lens)


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket %
                   total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(
                    len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * \
                (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j *
                                                           self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size


if __name__ == "__main__":
    import json

    class Te:
        def __init__(self, **dic):
            self.__dict__.update(dic)

    param = {"meta_file": "total.txt",
             "token_scp": "token.scp",
             "mel_scp": "mel16k.scp",
             }
    param = Te(**param)

    configPath = "stage2/vctk_config.json"
    with open(configPath, "rt", encoding="utf8") as f:
        data = f.read()
    config = json.loads(data)
    config = Te(**config)

    dataset = TokenVocoderDataset(param,
                                  config,
                                  "train")

    token_seq, segment_mels, data_norm, spec = dataset[0]
    print(token_seq.size())
    print(segment_mels.size())
    print(data_norm.size())
    print(spec.size())

    collateFn = TokenVocoderCollate(config.segment_size)
