import os
import numpy as np
import torch
import argparse
import logging
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
    parser.add_argument('--pause', type=int, required=False, default=1.5)
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
        pause_tmpt = np.array(input_ids)
        # 找的是符号的位置。 pause_mask中sp对应位置是0,其他位置是1.
        pause_mask = np.where(pause_tmpt == 2, 0, 1)
        # 找的是符号的位置。 pause_valu中sp对应位置是1,其他位置是0.
        # pause_valu = np.where(pause_tmpt == 2, 1, 0)
        pause_valu = 1 - pause_mask
        assert args.pause > 1
        pause_valu = pause_valu * ((args.pause * 16) // 256)
        with torch.inference_mode():
            x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
            x_tst_prosody = torch.FloatTensor(
                char_embeds).unsqueeze(0).to(device)
            pause_mask = torch.FloatTensor(
                pause_mask)[None, None, :].to(device)
            pause_valu = torch.FloatTensor(
                pause_valu)[None, None, :].to(device)
            audio = net_g.infer_pause(x_tst, x_tst_lengths, x_tst_prosody, pause_mask, pause_valu, noise_scale=0.5,
                                      length_scale=1)[0][0, 0].data.cpu().float().numpy()
        save_wav(
            audio, f"./vits_infer_out/bert_vits_{n}.wav", hps.data.sampling_rate)
    fo.close()
