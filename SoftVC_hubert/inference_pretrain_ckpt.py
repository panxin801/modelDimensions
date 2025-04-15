import torch
import torchaudio

if __name__ == "__main__":
    # Load checkpoint (either hubert_soft or hubert_discrete)
    type = "discrete"  # or "soft"
    hubert = torch.hub.load("bshall/hubert:main",
                            f"hubert_{type}", trust_repo=True).cuda()

    # Load audio
    wav, sr = torchaudio.load("testWavs/LJ001-000116k.wav")
    assert sr == 16000
    wav = wav.unsqueeze(0).cuda()  # [B,1,T] T=samples

    # Extract speech units
    with torch.inference_mode():
        # units.size()=[B,T,D], D=256, float32 tensor
        units = hubert.units(wav)
