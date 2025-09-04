"""Chunk version streaming inference
"""

import os
import numpy as np
import logging
import torch
import argparse
from scipy.io import wavfile

import utils
from text.symbols import symbols
from text import cleaned_text_to_sequence
from vits_pinyin import VITS_PinYin

logging.getLogger("numba").setLevel(logging.INFO)


def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Inference code for bert vits models')
    parser.add_argument('--config', type=str, required=False,
                        default="configs/bert_vits.json")
    parser.add_argument('--model', type=str, required=False,
                        default="vits_bert_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pinyin
    tts_front = VITS_PinYin("./bert", device)

    # config
    hps = utils.get_hparams_from_file(args.config)

    # model
    net_g = utils.load_class(hps.train.eval_class)(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)

    utils.load_model(args.model, net_g)
    net_g.eval()
    net_g.to(device)

    os.makedirs("./vits_infer_out/", exist_ok=True)

    n = 0
    fo = open("vits_infer_item.txt", "r+", encoding='utf-8')
    while (True):
        try:
            item = fo.readline().strip()
        except Exception as e:
            print('nothing of except:', e)
            break
        if (item == None or item == ""):
            break
        n = n + 1
        phonemes, char_embeds = tts_front.chinese_to_phonemes(item)
        input_ids = cleaned_text_to_sequence(phonemes)
        with torch.no_grad():
            x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
            x_tst_prosody = torch.FloatTensor(
                char_embeds).unsqueeze(0).to(device)
            audio = net_g.infer_stream(
                x_tst, x_tst_lengths, x_tst_prosody, noise_scale=0.5, length_scale=1)  # 分块生成，一起推理出来的。
        save_wav(
            audio, f"./vits_infer_out/bert_vits_stream_{n}.wav", hps.data.sampling_rate)
    fo.close()
