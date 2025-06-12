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


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0, is_bias=False,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=is_bias)
        self.ups.apply(commons.init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            nn.utils.remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = nn.utils.weight_norm if use_spectral_norm == False else nn.utils.spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(commons.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(commons.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(commons.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(commons.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(commons.get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(
            nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = nn.utils.weight_norm if use_spectral_norm == False else nn.utils.spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + \
            [DiscriminatorP(i, use_spectral_norm=use_spectral_norm)
             for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


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

        text_mask = torch.unsqueeze(commons.sequence_mask(
            text_lengths, text.size(1)), 1).to(y.dtype)
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


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        if g != None:
            g = g.detach()
        x_mask = torch.unsqueeze(commons.sequence_mask(
            x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


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


class SynthesizerTrn(nn.Module):
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
        version="v2",
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.version = version

        self.use_sdp = use_sdp
        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            version=version,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        # self.version=os.environ.get("version","v1")
        if self.version == "v1":
            self.ref_enc = modules.MelStyleEncoder(
                spec_channels, style_vector_dim=gin_channels)
        else:
            self.ref_enc = modules.MelStyleEncoder(
                704, style_vector_dim=gin_channels)

        ssl_dim = 768
        assert semantic_frame_rate in ["25hz", "50hz"]
        self.semantic_frame_rate = semantic_frame_rate
        if semantic_frame_rate == "25hz":
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.quantizer = quantize.ResidualVectorQuantizer(
            dimension=ssl_dim, n_q=1, bins=1024)
        self.freeze_quantizer = freeze_quantizer

    def forward(self, ssl, y, y_lengths, text, text_lengths):
        y_mask = torch.unsqueeze(commons.sequence_mask(
            y_lengths, y.size(2)), 1).to(y.dtype)
        if self.version == "v1":
            ge = self.ref_enc(y * y_mask, y_mask)
        else:
            ge = self.ref_enc(y[:, :704] * y_mask, y_mask)
        with autocast(enabled=False):
            maybe_no_grad = torch.no_grad() if self.freeze_quantizer else contextlib.nullcontext()
            with maybe_no_grad:
                if self.freeze_quantizer:
                    self.ssl_proj.eval()
                    self.quantizer.eval()
            ssl = self.ssl_proj(ssl)
            quantized, codes, commit_loss, quantized_list = self.quantizer(ssl, layers=[
                                                                           0])

        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(
                quantized.shape[-1] * 2), mode="nearest")

        x, m_p, logs_p, y_mask = self.enc_p(
            quantized, y_lengths, text, text_lengths, ge)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=ge)
        z_p = self.flow(z, y_mask, g=ge)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=ge)
        return (
            o,
            commit_loss,
            ids_slice,
            y_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            quantized,
        )

    def infer(self, ssl, y, y_lengths, text, text_lengths, test=None, noise_scale=0.5):
        y_mask = torch.unsqueeze(commons.sequence_mask(
            y_lengths, y.size(2)), 1).to(y.dtype)
        if self.version == "v1":
            ge = self.ref_enc(y * y_mask, y_mask)
        else:
            ge = self.ref_enc(y[:, :704] * y_mask, y_mask)

        ssl = self.ssl_proj(ssl)
        quantized, codes, commit_loss, _ = self.quantizer(ssl, layers=[0])
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(
                quantized.shape[-1] * 2), mode="nearest")

        x, m_p, logs_p, y_mask = self.enc_p(
            quantized, y_lengths, text, text_lengths, ge, test=test)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o, y_mask, (z, z_p, m_p, logs_p)

    @torch.no_grad()
    def decode(self, codes, text, refer, noise_scale=0.5, speed=1):
        def get_ge(refer):
            ge = None
            if refer is not None:
                refer_lengths = torch.LongTensor(
                    [refer.size(2)]).to(refer.device)
                refer_mask = torch.unsqueeze(commons.sequence_mask(
                    refer_lengths, refer.size(2)), 1).to(refer.dtype)
                if self.version == "v1":
                    ge = self.ref_enc(refer * refer_mask, refer_mask)
                else:
                    ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
            return ge

        if type(refer) == list:
            ges = []
            for _refer in refer:
                ge = get_ge(_refer)
                ges.append(ge)
            ge = torch.stack(ges, 0).mean(0)
        else:
            ge = get_ge(refer)

        y_lengths = torch.LongTensor([codes.size(2) * 2]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(
                quantized.shape[-1] * 2), mode="nearest")
        x, m_p, logs_p, y_mask = self.enc_p(
            quantized, y_lengths, text, text_lengths, ge, speed)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o

    def extract_latent(self, x):
        ssl = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)
        return codes.transpose(0, 1)


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
        b, _, t = x1.shape
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
            loss += self.criterion(vt_pred[i, :, prompt_lens[i]
                                   : x_lens[i]], vt[i, :, prompt_lens[i]: x_lens[i]])
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
        with autocast(device_type="cuda", enabled=False):
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


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import utils
    hps = utils.get_hparams(stage=2)

    ssl = torch.randn(1, 768, 160).cuda()
    spec = torch.randn(1, 1025, 160).cuda()
    mel = torch.randn(1, 100, 32).cuda()
    ssl_lengths = torch.LongTensor([158])
    spec_lengths = torch.LongTensor([158]).cuda()
    text = torch.randint(0, 100, (1, 25)).cuda()
    text_lengths = torch.LongTensor([25]).cuda()
    mel_lengths = torch.LongTensor([317]).cuda()
    use_grad_ckpt = False

    model = SynthesizerTrnV3(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    model.train().cuda()
    rets = model(ssl, spec, mel, ssl_lengths, spec_lengths,
                 text, text_lengths, mel_lengths, use_grad_ckpt=use_grad_ckpt)
    print(rets.shape)
