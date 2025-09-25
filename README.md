# modelDimensions
Personal read code detail

VITS: cVae+Flow e2e(end-to-end) TTS model.

HIFI-GAN: Non-autoregressive vocoder.

Gpt-Sovits: Few-shot Voice conversion and TTS system.

emotional-vits: Emotional vits model.

NT_GPT_TTS： Neural transducer two stage TTS model. Working like GPT_sovits.

VallE: First language model like TTS model, using neural codec extract discrete codes as intermediate and put into neural decoder synthesize wavs.




## Details
- vits_chinese: https://github.com/wac81/vits_chinese

- vits_chinese_stream -> vits_chinese with block streaming：https://github.com/PlayVoice/vits_chinese

    After checking, model arch of these two repo are the same, which means stream infering is a development modification. But in default `vits_chinese_stream using dp` and `vits_chinese using sdp`

    

- HiFi-GAN: https://github.com/jik876/hifi-gan

- soft_vc: https://github.com/bshall/soft-vc
    - soft_vc_hubert: https://github.com/bshall/hubert
    - soft_vc_aouctic: https://github.com/bshall/acoustic-model
    
- whisper-vits-svc: https://github.com/PlayVoice/whisper-vits-svc/tree/bigvgan-mix-v2

- bert_vits2: https://github.com/fishaudio/Bert-VITS2
  
    - vits2: https://github.com/p0p4k/vits2_pytorch/tree/main
    
- gpt_sovits: https://github.com/RVC-Boss/GPT-SoVITS

- emotional_vits: https://github.com/innnky/emotional-vits

- NT_GPT_TTS(unoffical implementation): https://github.com/scutcsq/Neural-Transducers-for-Two-Stage-Text-to-Speech-via-Semantic-Token-Prediction?tab=readme-ov-file
    - paper: https://arxiv.org/pdf/2401.01498 

    Using wav2vec2 and K-Means extract discrete token as stage1,  put into vits architecture acoustic as stage2, pipeline similar to GPT_VITS
    
- VallE(unoffical implementation): https://github.com/lifeiteng/vall-e

