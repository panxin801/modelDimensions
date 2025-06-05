import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import random
import contextlib
import warnings
warnings.filterwarnings("ignore")
from torch.amp import autocast


from module import (commons, modules, attentions, quantize)
from module.mrte_model import MRTE
from text import symbols as symbols_v1
from text import symbols2 as symbols_v2
from f5_tts.model import DiT


class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        latent_channels=192,
        version="v2",
    ):
        super().__init__()

        self.out_channels = out_channels  # 192
        self.hidden_channels = hidden_channels  # 192
        self.filter_channels = filter_channels  # 768
        self.n_heads = n_heads  # 2
        self.n_layers = n_layers  # 6
        self.kernel_size = kernel_size  # 3
        self.p_dropout = p_dropout  # 0.1
        self.latent_channels = latent_channels  # 192
        self.version = version  # "v2"

        self.ssl_proj = nn.Conv1d(768, hidden_channels, 1)

        self.encoder_ssl = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )
        self.encoder_text = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )

        if self.version == "v1":
            symbols = symbols_v1.symbols
        else:
            symbols = symbols_v2.symbols
        self.text_embedding = nn.Embedding(len(symbols), hidden_channels)

        self.mrte = MRTE()

        self.encoder2 = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, y, y_lengths, text, text_lengths, ge, speed=1, test=None):
        y_mask = torch.unsqueeze(commons.sequence_mask(
            y_lengths, y.size(2)), 1).to(y.dtype)

        y = self.ssl_proj(y * y_mask) * y_mask
        y = self.encoder_ssl(y * y_mask, y_mask)

        text_mask = torch.unsqueeaze(commons.sequence_mask(
            text_lengths, text.size(1)), 1).tO(y.dtype)
        if test == 1:
            text[:, :] = 0
        text = self.text_embedding(text).transpose(1, 2)
        text = self.encoder_text(text * text_mask, text_mask)
        y = self.mrte(y, y_mask, text, text_mask, ge)
        y = self.encoder2(y * y_mask, y_mask)
        if speed != 1:
            y = F.interpolat(y, size=int(
                y.size(-1) / speed) + 1, mode="linear")
            y_mas = F.interpolate(y_mask, size=y.size(-1), mode="nearest")
        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask

    def extract_latent(self, x):
        x = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x)
        return codes.transpose(0, 1)

    def decode_latent(self, codes, y_mask, refer, refer_mask, ge):
        quantized = self.quantizer.decode(codes)

        y = self.vq_proj(quantized) * y_mask
        y = self.encoder_ssl(y * y_mask, y_mask)

        y = self.mrte(y, y_mask, refer, refer_mask, ge)
        y = self.encoder2(y * y_mask, y_mask)

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask, quantized


class Encoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0
    ):
        super().__init__()
        self.in_channels = in_channels  # 512
        self.out_channels = out_channels  # 512
        self.hidden_channels = hidden_channels  # 512
        self.kernel_size = kernel_size  # 5
        self.dilation_rate = dilation_rate  # 1
        self.n_layers = n_layers  # 8
        self.gin_channels = gin_channels  # 512

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size,
                              dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_lengths, g=None):
        if g != None:
            g = g.detach()
        x_mask = torch.unsqueeze(commons.sequence_mask(
            x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        return stats, x_mask


class CFM(nn.Module):
    def __init__(self, in_channels, dit):
        super().__init__()
        self.estimator = dit

        self.sigma_min = 1e-6
        self.in_channels = in_channels  # 100
        self.criterion = nn.MSELoss()

    @torch.inference_mode()
    def inference(self,
                  mu,
                  x_lens,
                  prompt,
                  n_timesteps,
                  temperature=1.0,
                  inference_cfg_rate=0):
        """Forward diffusion"""
        B, T = mu.size(0), mu.size(1)
        x = torch.randn([B, self.in_channels, T],
                        device=mu.device, dtype=mu.dtype) * temperature
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x, dtype=mu.dtype)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        mu = mu.transpose(2, 1)
        t = 0
        d = 1 / n_timesteps
        for j in range(n_timesteps):
            t_tensor = torch.ones(
                x.size(0), device=x.device, dtype=mu.dtype) * t
            d_tensor = torch.ones(
                x.size(0), device=x.device, dtype=mu.dtype) * d
            v_pred = self.estimator(
                x, prompt_x, x_lens, t_tensor, d_tensor, mu, use_grad_ckpt=False, drop_audio_cond=False, drop_text=False).transpose(2, 1)
            if inference_cfg_rate > 1e-5:
                neg = self.estimator(
                    x,
                    prompt_x,
                    x_lens,
                    t_tensor,
                    d_tensor,
                    mu,
                    use_grad_ckpt=False,
                    drop_audio_cond=True,
                    drop_text=True
                ).transpose(2, 1)
                v_pred = v_pred + (v_pred - neg) * inference_cfg_rate
            x = x + d * v_pred
            t = t + d
            x[:, :, :prompt_len] = 0
        return x

    def forward(self, x1, x_lens, prompt_lens, mu, use_grad_ckpt):
        b, _, t = x1.shape()
        t = torch.rand([b], device=mu.device, dtype=x1.dtype)
        x0 = torch.randn_like(x1, device=mu.device)
        vt = x1 - x0
        xt = x0 + t[:, None, None] * vt
        dt = torch.zeros_like(t, device=mu.device)
        prompt = torch.zeros_like(x1)
        for i in range(b):
            prompt[i, :, : prompt_lens[i]] = x1[i, :, : prompt_lens[i]]
            xt[i, :, : prompt_lens[i]] = 0
        gailv = 0.3  # if ttime()>1736250488 else 0.1
        if random.random() < gailv:
            base = torch.randint(2, 8, (t.shape[0],), device=mu.device)
            d = 1 / torch.pow(2, base)
            d_input = d.clone()
            d_input[d_input < 1e-2] = 0
            # with torch.no_grad():
            v_pred_1 = self.estimator(
                xt, prompt, x_lens, t, d_input, mu, use_grad_ckpt).transpose(2, 1).detach()
            # v_pred_1 = self.diffusion(xt, t, d_input, cond=conditioning).detach()
            x_mid = xt + d[:, None, None] * v_pred_1
            # v_pred_2 = self.diffusion(x_mid, t+d, d_input, cond=conditioning).detach()
            v_pred_2 = self.estimator(
                x_mid, prompt, x_lens, t + d, d_input, mu, use_grad_ckpt).transpose(2, 1).detach()
            vt = (v_pred_1 + v_pred_2) / 2
            vt = vt.detach()
            dt = 2 * d

        vt_pred = self.estimator(
            xt, prompt, x_lens, t, dt, mu, use_grad_ckpt).transpose(2, 1)
        loss = 0
        for i in range(b):
            loss += self.criterion(vt_pred[i, :, prompt_lens[i]                                   : x_lens[i]], vt[i, :, prompt_lens[i]: x_lens[i]])
        loss /= b

        return loss


def set_no_grad(net_g):
    for name, param in net_g.named_parameters():
        param.requires_grad = False


class SynthesizerTrnV3(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        semantic_frame_rate=None,
        freeze_quantizer=None,
        version="v3",
        **kwargs,
    ):
        super().__init__()

        self.spec_channels = spec_channels  # 1025
        self.inter_channels = inter_channels  # 192
        self.hidden_channels = hidden_channels  # 192
        self.filter_channels = filter_channels  # 768
        self.n_heads = n_heads  # 2
        self.n_layers = n_layers  # 6
        self.kernel_size = kernel_size  # 3
        self.p_dropout = p_dropout  # 0.1
        self.resblock = resblock  # "1"
        self.resblock_kernel_sizes = resblock_kernel_sizes  # [3,7,11]
        # [[1,3,5],[1,3,5],[1,3,5]]
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates  # [10,8,8,2,2]
        self.upsample_initial_channel = upsample_initial_channel  # 512
        self.upsample_kernel_sizes = upsample_kernel_sizes  # [16,16,8,2,2]
        self.segment_size = segment_size  # 32
        self.n_speakers = n_speakers  # 300
        self.gin_channels = gin_channels  # 512
        self.version = version  # "v3"

        self.model_dim = 512
        self.use_sdp = use_sdp  # True
        self.enc_p = TextEncoder(inter_channels, hidden_channels,
                                 filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        # self.ref_enc = modules.MelStyleEncoder(spec_channels, style_vector_dim=gin_channels)###Rollback
        self.ref_enc = modules.MelStyleEncoder(
            704, style_vector_dim=gin_channels)  # Rollback
        # self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
        #                      upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        # self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16,
        #                               gin_channels=gin_channels)
        # self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        ssl_dim = 768
        assert semantic_frame_rate in ["25hz", "50hz"]  # "25hz"
        self.semantic_frame_rate = semantic_frame_rate
        if semantic_frame_rate == "25hz":
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.quantizer = quantize.ResidualVectorQuantizer(
            dimension=ssl_dim, n_q=1, bins=1024)
        self.freeze_quantizer = freeze_quantizer  # True
        inter_channels2 = 512
        self.bridge = nn.Sequential(
            nn.Conv1d(inter_channels, inter_channels2, 1, stride=1),
            nn.LeakyReLU()
        )
        self.wns1 = Encoder(inter_channels2, inter_channels2,
                            inter_channels2, 5, 1, 8, gin_channels=gin_channels)
        self.linear_mel = nn.Conv1d(inter_channels2, 100, 1, stride=1)
        self.cfm = CFM(100, DiT(**dict(dim=1024,
                                       depth=22,
                                       heads=16,
                                       ff_mult=2,
                                       text_dim=inter_channels2,
                                       conv_layers=4)
                                ),
                       )  # text_dim is condition feature dim
        if self.freeze_quantizer:
            set_no_grad(self.ssl_proj)
            set_no_grad(self.quantizer)
            set_no_grad(self.enc_p)

    def forward(
        self, ssl, y, mel, ssl_lengths, y_lengths, text, text_lengths, mel_lengths, use_grad_ckpt
    ):  # ssl_lengths no need now
        with autocast(enabled=False):
            y_mask = torch.unsqueeze(commons.sequence_mask(
                y_lengths, y.size(2)), 1).to(y.dtype)
            ge = self.ref_enc(y[:, :704] * y_mask, y_mask)
            maybe_no_grad = torch.no_grad() if self.freeze_quantizer else contextlib.nullcontext()
            with maybe_no_grad:
                if self.freeze_quantizer:
                    self.ssl_proj.eval()
                    self.quantizer.eval()
                    self.enc_p.eval()
                ssl = self.ssl_proj(ssl)
                quantized, codes, commit_loss, quantized_list = self.quantizer(ssl, layers=[
                                                                               0])
                quantized = F.interpolate(
                    quantized, scale_factor=2, mode="nearest")  # BCT
                x, m_p, logs_p, y_mask = self.enc_p(
                    quantized, y_lengths, text, text_lengths, ge)

        fea = self.bridge(x)
        fea = F.interpolate(fea, scale_factor=(
            1.875 if self.version == "v3" else 2), mode="nearest")  # BCT
        # If the 1-minute fine-tuning works fine, no need to manually adjust the learning rate.
        fea, y_mask_ = self.wns1(fea, mel_lengths, ge)
        B = ssl.size(0)
        prompt_len_max = mel_lengths * 2 / 3
        prompt_len = (torch.rand([B], device=fea.device)
                      * prompt_len_max).floor().to(dtype=torch.long)
        minn = min(mel.size(-1), fea.size(-1))
        mel = mel[:, :, :minn]
        fea = fea[:, :, :minn]
        cfm_loss = self.cfm(mel, mel_lengths, prompt_len, fea, use_grad_ckpt)
        return cfm_loss

    @torch.inference_mode()
    def decode_encp(self, codes, text, refer, ge=None, speed=1):
        if ge is None:
            refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
            refer_mask = torch.unsqueeze(commons.sequence_mask(
                refer_lengths, refer.size(2)), 1).to(refer.dtype)
            ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
        y_lengths = torch.LongTensor([int(codes.size(2) * 2)]).to(codes.device)
        if speed == 1:
            sizes = int(codes.size(2) * (3.875 if self.version == "v3"else 4))
        else:
            sizee = int(codes.size(2) *
                        (3.875 if self.version == "v3"else 4) / speed) + 1
        y_lengths1 = torch.LongTensor([sizes]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).tO(text.device)

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(
                quantized, scale_factor=2, mode="nearest")  # BCT
        x, m_p, los_p, y_mask = self.enc_p(
            quantized, y_lengths, text, text_lengths, ge, speed)
        fea = self.bridge(x)
        fea = F.interpolate(fea, scale_factor=(
            1.875 if self.version == "v3"else 2), mode="nearest")  # BCT
        # more wn paramter to learn mel
        fea, y_mask_ = self.wns1(fea, y_lengths1, ge)
        return fea, ge

    def extract_latent(self, x):
        """
        x: [B,D=768,F], cnhubert extract from 32k wavs
        return: 
        codes.transpose=[D,B,F/2]
        """
        ssl = self.ssl_proj(x)  # [b,D,F/2]
        quantized, codes, commit_loss, quantized_list = self.quantizer(
            ssl)  # codes 是残差后的ssl index, quantized 是残差后的ssl rvq 码本值
        return codes.transpose(0, 1)
