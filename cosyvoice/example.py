import sys
sys.path.append("third_party/Matcha-TTS")
import torchaudio

from cosyvoice.cli.cosyvoice import AutoModel


def cosyvoice_example():
    """ CosyVoice Usage, check https://fun-audio-llm.github.io/ for more details
    """

    cosyvoice = AutoModel(
        model_dir="/data/cosyvoicePretrainModels/pretrained_models/CosyVoice-300M-SFT")
    # sft usage
    print(cosyvoice.list_available_spks())
    # change stream=True for chunk stream inference
    for i, j in enumerate(cosyvoice.inference_sft("你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？", "中文女", stream=True)):
        torchaudio.save(f"sft_{i}.wav", j["tts_speech"], cosyvoice.sample_rate)

    cosyvoice = AutoModel(
        model_dir="/data/cosyvoicePretrainModels/pretrained_models/CosyVoice-300M")
    # zero_shot usage
    for i, j in enumerate(cosyvoice.inference_zero_shot("收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。", "希望你以后能够做的比我还好呦。", "asset\zero_shot_prompt.wav")):
        torchaudio.save(f"zero_shot_{i}.wav",
                        j["tts_speech"], cosyvoice.sample_rate)
    # cross_lingual usage, <|zh|><|en|><|ja|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
    for i, j in enumerate(cosyvoice.inference_cross_lingual("< |en | >And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.",
                                                            "./asset/cross_lingual_prompt.wav")):
        torchaudio.save(f"cross_lingual_{i}.wav",
                        j["tts_speech"], cosyvoice.sample_rate)
    # vc usage
    for i, j in enumerate(cosyvoice.inference_vc("./asset/cross_lingual_prompt.wav", "./asset/zero_shot_prompt.wav")):
        torchaudio.save(f"vc_{i}.wav", j["tts_speech"], cosyvoice.sample_rate)

    cosyvoice = AutoModel(
        model_dir="/data/cosyvoicePretrainModels/pretrained_models/CosyVoice-300M-Instruct")
    # instruct usage, support <laughter></laughter><strong></strong>[laughter][breath]
    for i, j in enumerate(cosyvoice.inference_instruct(f"在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。", "中文男", "'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.<|endofprompt|>")):
        torchaudio.save(f"instruct_{i}.wav",
                        j["tts_speech"], cosyvoice.sample_rate)


def main():
    cosyvoice_example()


if __name__ == "__main__":
    main()
