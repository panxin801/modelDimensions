import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile

import utils
import commons
from text import text_to_sequence
from text.symbols import symbols
from models import SynthesizerTrn


def get_text(text, hps):
    # type(text_norm)=list, len(text_norm)=99
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        # 在文本开头增加一个空白符，每个音素后面增加一个空白符
        # len(text_norm)=199
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)  # text_norm.shape=[199]

    return text_norm


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hps = utils.get_hparams_from_file("./configs/woman_csmsc.json")
    # stn_tst 是文本转换为symbol的数字序列。长度为音素数量
    stn_tst = get_text("第一，南京不是发展的不行，是大家对他期望很高。", hps)

    net_g = SynthesizerTrn(len(symbols),  # 178
                           hps.data.filter_length // 2 + 1,  # 1024//2+1=513
                           hps.train.segment_size // hps.data.hop_length,  # 8192//256=32
                           **hps.model).to(device)
    net_g.eval()
    _ = utils.load_checkpoint("./logs/woman_csmsc/G_100000.pth", net_g, None)

    with torch.no_grad():
        # Prepare data
        x_tst = stn_tst.to(device).unsqueeze(0)  # x_tst.size()=[1, 199]
        x_tst_lengths = torch.LongTensor([x_tst.size(1)]).to(
            device)  # x_tst_lengths.size()=[1]

        audio = net_g.infer(
            x_tst, x_tst_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1)[0][0, 0].data.cpu().float().numpy()

    sample_rate = 24000
    wavfile.write("abc1.wav", sample_rate, audio)
