# Copy from https://github.com/bshall/soft-vc

import torch
import torchaudio

if __name__ == "__main__":
    # Load the content encoder (either hubert_soft or hubert_discrete)
    hubert = torch.hub.load("bshall/hubert:main",
                            "hubert_soft", trust_repo=True).cuda()

    # Load the acoustic model (either hubert_soft or hubert_discrete)
    acoustic = torch.hub.load("bshall/acoustic-model:main",
                              "hubert_soft", trust_repo=True).cuda()

    # Load the vocoder (either hifigan_hubert_soft or hifigan_hubert_discrete)
    hifigan = torch.hub.load("bshall/hifigan:main",
                             "hifigan_hubert_soft", trust_repo=True).cuda()

    # Load the source audio
    source, sr = torchaudio.load("testWavs/LJ001-006816k.wav")
    assert sr == 16000
    source = source.unsqueeze(0).cuda()

    # Convert to the target speaker
    with torch.inference_mode():
        # Extract speech units
        units = hubert.units(source)
        # Generate target spectrogram
        mel = acoustic.generate(units).transpose(1, 2)
        # Generate audio waveform
        target = hifigan(mel)
