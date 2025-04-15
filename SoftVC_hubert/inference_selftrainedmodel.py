import torch
import torchaudio

from hubert.model import (Hubert, HubertSoft)

if __name__ == "__main__":
    # Load checkpoint (either hubert_soft or hubert_discrete)
    ckpt = "outputs/model-10.pt"
    stateDict = torch.load(ckpt, map_location="cpu")
    newstateDict = {}
    for k, v in stateDict["hubert"].items():
        if k.startswith("module."):
            eles = k.split(".", 1)
            newstateDict[".".join(eles[1:])] = v
        else:
            newstateDict[k] = v

    hubert = HubertSoft()
    hubert.load_state_dict(newstateDict)
    hubert.cuda().eval()

    # Load audio
    wav, sr = torchaudio.load("testWavs/LJ001-000116k.wav")
    assert sr == 16000
    wav = wav.unsqueeze(0).cuda()  # [B,1,T] T=samples

    # Extract speech units
    with torch.inference_mode():
        # units.size()=[B,T,D], D=256, float32 tensor
        units = hubert.units(wav)
