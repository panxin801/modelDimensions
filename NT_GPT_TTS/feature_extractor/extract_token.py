import joblib
import numpy as np
import kaldiio
import os

if __name__ == "__main__":
    readDir = "vctkDataset/wav2vec2_15th"
    saveDir = "vctkDataset/token_wav2vec2_15th"
    modelPath = "km.bin"

    # Load KMeans model
    model = joblib.load(open(modelPath, "rb"))

    os.makedirs(saveDir, exist_ok=True)

    total_token_seq = {}
    for file in os.listdir(readDir):
        baseName = os.path.splitext(file)[0]
        readPath = os.path.join(readDir, file)

        feats = np.load(readPath)  # [1, num_frame, 1024]
        print(f"feats shape: {feats.shape}")

        # quantize feats named as tokens
        tokens = model.predict(feats[0])  # [num_frame]

        # save tokens
        tokenSavePath = os.path.join(saveDir, file)
        np.save(tokenSavePath, tokens)

        total_token_seq[baseName] = tokens

    kaldiio.save_ark("token.ark", total_token_seq, "token.scp", False)
