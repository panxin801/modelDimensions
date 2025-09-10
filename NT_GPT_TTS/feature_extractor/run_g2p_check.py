# g2p english words into phonemes

import g2p_en
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1.phone import symbols


if __name__ == "__main__":
    readFile = "mel16k.scp"
    txtDir = "vctkDataset/txt"

    saveFile = "total.txt"
    fw = open(saveFile, "wt", encoding="utf8")

    # Init g2p model
    g2p = g2p_en.G2p()

    # Read scp file
    for line in open(readFile, "rt", encoding="utf8"):
        uttid, content = line.strip().split(" ")
        spk = uttid.split("_", 1)[0]
        baseName = uttid.rsplit("_", 1)[0]
        txtFile = os.path.join(txtDir, spk, baseName + ".txt")

        with open(txtFile, "rt", encoding="utf8") as fr:
            text = fr.read().strip()
            phonemes = g2p(text)
            phonemes = [ele if ele != " " else "sil" for ele in phonemes]

            # 处理.
            if phonemes[-1] == ".":
                phonemes[-1] = "<EOS>"
            else:
                phonemes.append("<EOS>")

            # 增加BOS
            phonemes.insert(0, "<BOS>")

            # deal with symbols
            for idx, ele in enumerate(phonemes):
                if ele in [",", "?", "!", "-"]:
                    phonemes[idx] = "sil"

            for ele in phonemes:
                if not ele in symbols:
                    print(ele)
                    print(uttid)
                    sys.exit(1)
