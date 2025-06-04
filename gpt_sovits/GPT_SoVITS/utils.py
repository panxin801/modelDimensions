import librosa
import torch
import torchaudio
import json


def load_wav_to_torch_and_resample(file_path: str, target_sample_rate: int = 16000) -> torch.Tensor:
    # 加载音频文件
    waveform, sample_rate = librosa.load(file_path, sr=None)

    # 如果采样率与目标采样率不同，则进行重采样
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(torch.from_numpy(waveform))

    # 返回音频张量
    return waveform


def load_wav_to_torch(full_path):
    data, sampling_rate = librosa.load(full_path, sr=None)
    return torch.FloatTensor(data), sampling_rate


def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


if __name__ == "__main__":
    print(
        load_wav_to_torch(
            "/home/fish/wenetspeech/dataset_vq/Y0000022499_wHFSeHEx9CM/S00261.flac",
        )
    )
