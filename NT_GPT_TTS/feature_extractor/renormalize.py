import os
import subprocess

if __name__ == "__main__":
    readDir = "vctkDataset/wav48_silence_trimmed"
    outDir = "vctkDataset/totalwav"
    os.makedirs(outDir, exist_ok=True)

    for spkDir in os.listdir(readDir):
        if os.path.isdir(os.path.join(readDir, spkDir)):
            tmpDir = os.path.join(os.path.join(readDir, spkDir))

            for file in os.listdir(tmpDir):
                readPath = os.path.join(tmpDir, file)
                if os.path.isfile(readPath) and file.endswith(".flac"):
                    saveName = os.path.join(
                        outDir, file.replace(".flac", ".wav"))
                    cmd = f"ffmpeg -i {readPath} -ac 1 -ar 16000 -acodec pcm_s16le {saveName}"
                    subprocess.call(cmd, shell=True)
