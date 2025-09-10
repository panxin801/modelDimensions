import os
import kaldiio
import numpy as np
import torch
import librosa
from librosa.filters import mel as librosa_mel_fn
from scipy.io import wavfile

MAX_WAV_VALUE = 32768.0
mel_basis = {}
hann_window = {}


def load_wav(full_path):
    sr, data = wavfile.read(full_path)
    return data, sr


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft,
                             n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + '_' + str(y.device)
                  ] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(
        1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def extract_mel(readDir, saveDir, dataset):
    os.makedirs(saveDir, exist_ok=True)

    sampling_rate = 16000
    n_fft = 1024
    num_mels = 80
    hop_size = 256
    win_size = 1024
    fmin = 0
    fmax = 8000

    for file in os.listdir(readDir):
        if (os.path.exists(os.path.join(readDir, file).replace(".wav", ".npy"))):
            continue

        # VCTK
        if dataset.lower() == "vctk":
            if file.endswith("mic1.wav"):
                audio, sr = load_wav(os.path.join(readDir, file))

                if (sr != sampling_rate):
                    print(f"{file} {sr=}, need {sampling_rate}")
                else:
                    audio = audio / MAX_WAV_VALUE
                    audio = librosa.util.normalize(audio) * 0.95
                    audio = torch.FloatTensor(audio).unsqueeze(0)
                    mel = mel_spectrogram(
                        audio, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False)
                    mel = mel.permute(0, 2, 1)
                    mel = mel.squeeze(0)
                    mel = mel.data.numpy()
                    np.save(os.path.join(
                        saveDir, file.replace(".wav", ".npy")), mel)

        # LibriTTS
        elif dataset.lower() == "libritts":
            if file.endswith(".wav"):
                audio, sr = load_wav(os.path.join(readDir, file))

                if (sr != sampling_rate):
                    print(f"{file} {sr=}, need {sampling_rate}")
                else:
                    audio = audio / MAX_WAV_VALUE
                    audio = librosa.util.normalize(audio) * 0.95
                    audio = torch.FloatTensor(audio).unsqueeze(0)
                    mel = mel_spectrogram(
                        audio, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False)
                    mel = mel.permute(0, 2, 1)
                    mel = mel.squeeze(0)
                    mel = mel.data.numpy()
                    np.save(os.path.join(
                        saveDir, file.replace(".wav", ".npy")), mel)


def mel2ark(saveDir):
    mels = {}

    for file in os.listdir(saveDir):
        mel = np.load(os.path.join(saveDir, file))
        mels[os.path.splitext(file)[0]] = mel
    kaldiio.save_ark("mel16k.ark", mels, "mel16k.scp", False)


if __name__ == "__main__":
    readDir = "vctkDataset/totalwav"  # 16k wav dir
    saveDir = "vctkDataset/totalmel"  # 16k wavs mel-spec dir

    extract_mel(readDir, saveDir, "vctk")
    mel2ark(saveDir)
