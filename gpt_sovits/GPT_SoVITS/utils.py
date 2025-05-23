import librosa
import torch
import torchaudio


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
