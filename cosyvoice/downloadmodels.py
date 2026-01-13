# modelscope SDK model download
from modelscope import snapshot_download

snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='/data/cosyvoicePretrainModels/pretrained_models/Fun-CosyVoice3-0.5B')  # finish
snapshot_download('iic/CosyVoice2-0.5B',
                  local_dir='/data/cosyvoicePretrainModels/pretrained_models/CosyVoice2-0.5B')  # finish
snapshot_download('iic/CosyVoice-300M',
                  local_dir='/data/cosyvoicePretrainModels/pretrained_models/CosyVoice-300M')  # finish
snapshot_download('iic/CosyVoice-300M-SFT',
                  local_dir='/data/cosyvoicePretrainModels/pretrained_models/CosyVoice-300M-SFT')  # finish
snapshot_download('iic/CosyVoice-300M-Instruct',
                  local_dir='/data/cosyvoicePretrainModels/pretrained_models/CosyVoice-300M-Instruct')  # finish
snapshot_download('iic/CosyVoice-ttsfrd',
                  local_dir='/data/cosyvoicePretrainModels/pretrained_models/CosyVoice-ttsfrd')  # finish
