import torchaudio
import shutil
import json
from pathlib import Path

if __name__ == "__main__":
    readDir = Path("LJSpeech-1.1/wavs")
    trainSave = readDir / "train-11"
    devSave = readDir / "dev-21"
    saveJson = {}

    trainSave.mkdir(exist_ok=True, parents=True)
    devSave.mkdir(parents=True, exist_ok=True)

    with open("LJSpeech-1.1/validation.txt", "rt", encoding="utf8") as fr:
        devList = [line.strip().split("|", 1)[0] for line in fr]

    for line in devList:
        readName = readDir / (line + ".wav")
        data, sr = torchaudio.load(readName)
        saveName = devSave / readName.name
        saveJson[f"{str(saveName.parent)}/{saveName.stem}"] = data.size(1)
        # mv from readName to saveName
        shutil.move(readName, saveName)

    with open("LJSpeech-1.1/training.txt", "rt", encoding="utf8") as fr:
        trainList = [line.strip().split("|", 1)[0] for line in fr]

    for line in trainList:
        readName = readDir / (line + ".wav")
        data, sr = torchaudio.load(readName)
        saveName = trainSave / readName.name
        saveJson[f"{str(saveName.parent)}/{saveName.stem}"] = data.size(1)
        # mv from readName to saveName
        shutil.move(readName, saveName)

    # save saveJson to json file
    with open("LJSpeech-1.1/lengths.json", "wt", encoding="utf8") as fw:
        json.dump(saveJson, fw, indent=4)
