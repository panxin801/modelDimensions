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


def main():
    cosyvoice_example()


if __name__ == "__main__":
    main()
