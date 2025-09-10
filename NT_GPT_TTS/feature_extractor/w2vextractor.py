import torch
import os
import numpy as np
import soundfile as sf
from transformers import (Wav2Vec2Processor, Wav2Vec2ForCTC)

if __name__ == "__main__":
    ckptPath = "./wav2vec2-xlsr-53-espeak-cv-ft/"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    readDir = "vctkDataset/totalwav"
    saveDir = "vctkDataset/wav2vec2_15th"
    selected = 15
    os.makedirs(saveDir, exist_ok=True)

    processor = Wav2Vec2Processor.from_pretrained(ckptPath)
    model = Wav2Vec2ForCTC.from_pretrained(ckptPath).to(device)
    # Set evaluation mode
    model.eval()

    for wavPath in os.listdir(readDir):
        if wavPath.endswith("mic1.wav"):
            data, sr = sf.read(os.path.join(readDir, wavPath))
            inputValues = processor(
                data, sampling_rate=sr, return_tensors="pt").input_values.to(device)  # [1,Len]

            with torch.inference_mode():
                output = model(inputValues, output_hidden_states=True)
                hiddenStates = output.hidden_states[1]  # [1,T,Dim=1024]
            output = hiddenStates.data.cpu().numpy()
            saveName = os.path.join(saveDir, wavPath.replace(".wav", ".npy"))
            np.save(saveName, output)
