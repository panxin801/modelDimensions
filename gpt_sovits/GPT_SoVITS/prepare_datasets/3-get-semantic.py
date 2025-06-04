import os
import sys
import torch
import traceback
import logging

# Add path
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import utils
from tools import dataPrep_utils

logging.getLogger("numba").setLevel(logging.WARNING)


# from config import pretrained_s2G

# inp_text=sys.argv[1]
# exp_name=sys.argv[2]
# i_part=sys.argv[3]
# all_parts=sys.argv[4]
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[5]
# opt_dir="/data/docker/liujing04/gpt-vits/fine_tune_dataset/%s"%exp_name

def name2go(wav_name, lines):
    hubert_path = f"{hubert_dir}/{wav_name}.pt"
    if not os.path.exists(hubert_path):
        return

    ssl_content = torch.load(hubert_path, map_location="cpu")
    if is_half:
        ssl_content = ssl_content.half().to(device)
    else:
        ssl_content = ssl_content.to(device)

    with torch.inference_mode():
        codes = vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
    lines.append(f"{wav_name}\t{semantic}")


if __name__ == "__main__":
    # config = {'inp_text': 'data\\train.list', 'exp_name': 'TestA1', 'opt_dir': 'logs\\TestA1', 'pretrained_s2G': 'GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth',
    #           's2config_path': 'GPT_SoVITS/configs/s2.json', 'is_half': 'False', 'i_part': '0', 'all_parts': '2', '_CUDA_VISIBLE_DEVICES': '0'}
    # os.environ.update(config)

    inp_text = os.environ.get("inp_text")
    exp_name = os.environ.get("exp_name")
    i_part = os.environ.get("i_part")
    all_parts = os.environ.get("all_parts")
    if "_CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

    opt_dir = os.environ.get("opt_dir")
    pretrained_s2G = os.environ.get("pretrained_s2G")
    s2config_path = os.environ.get("s2config_path")

    if not os.path.exists(pretrained_s2G):
        raise FileNotFoundError(pretrained_s2G)

    size = os.path.getsize(pretrained_s2G)
    if size < 82978 * 1024:
        version = "v1"
    elif size < 100 * 1024 * 1024:
        version = "v2"
    elif size < 103520 * 1024:
        version = "v1"
    elif size < 700 * 1024 * 1024:
        version = "v2"
    else:
        version = "v3"

    is_half = eval(os.environ.get("is_half", "True")
                   ) and torch.cuda.is_available()

    if version != "v3":
        from module.models import SynthesizerTrn
    else:
        from module.models import SynthesizerTrnV3 as SynthesizerTrn

    hubert_dir = f"{opt_dir}/4-cnhubert"
    semantic_path = f"{opt_dir}/6-name2semantic-{i_part}.tsv"
    if not os.path.exists(semantic_path):
        os.makedirs(opt_dir, exist_ok=True)

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        hps = utils.get_hparams_from_file(s2config_path)
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            version=version,
            **hps.model
        )
        if is_half == True:
            vq_model = vq_model.half().to(device)
        else:
            vq_model = vq_model.to(device)
        vq_model.eval()

        print(vq_model.load_state_dict(torch.load(pretrained_s2G,
              map_location="cpu", weights_only=False)["weight"], strict=False))

        with open(inp_text, "rt", encoding="utf8") as fr:
            lines = fr.read().strip("\n").split("\n")

        lines1 = []
        for line in lines[int(i_part)::int(all_parts)]:
            try:
                wav_name, spk_name, language, text = line.split("|")
                wav_name = dataPrep_utils.clean_path(wav_name)
                # wav_name = os.path.basename(wav_name)
                wav_name = os.path.sep.join(wav_name.split(os.path.sep)[1:])
                name2go(wav_name, lines1)
            except:
                print(line, traceback.format_exc())
        with open(semantic_path, "wt", encoding="utf8") as f:
            f.write("\n".join(lines1))
