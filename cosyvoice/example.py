import sys
sys.path.append("third_party/Matcha-TTS")
import torchaudio

from cosyvoice.cli.cosyvoice import AutoModel


def cosyvoice_example():
    """ CosyVoice Usage, check https://fun-audio-llm.github.io/ for more details
    """

    cosyvoice = AutoModel(
        model_dir="cosyvoicePretrainModels\pretrained_models\CosyVoice-300M-SFT")
    # sft usage
    print(cosyvoice.list_available_spks())
    # change stream=True for chunk stream inference
    for i, j in enumerate(cosyvoice.inference_sft("你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？", "中文女", stream=False)):
        torchaudio.save(f"sft_{i}.wav", j["tts_speech"], cosyvoice.sample_rate)


def main():
    cosyvoice_example()


if __name__ == "__main__":
    main()
