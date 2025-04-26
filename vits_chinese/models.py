import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import attentions
import commons
import modules
import monotonic_align
from commons import (get_padding)


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
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


class SynthesizerTrn(nn.Module):
    def __init__(self,
                 n_vocab,
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
                 **kwargs):
        super().__init__()
        self.n_vocab = n_vocab  # 178
        self.spec_channels = spec_channels  # 513
        self.segment_size = segment_size  # 32
        self.inter_channels = inter_channels  # 192
        self.hidden_channels = hidden_channels  # 192
        self.filter_channels = filter_channels  # 768
        self.n_heads = n_heads  # 2
        self.n_layers = n_layers  # 6
        self.kernel_size = kernel_size  # 3
        self.p_dropout = p_dropout  # 0.1
        self.resblock = resblock  # "1"
        self.resblock_kernel_sizes = resblock_kernel_sizes  # [3,7,11]
        # resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]]
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates  # [8,8,2,2]
        self.upsample_initial_channel = upsample_initial_channel  # 512
        self.upsample_kernel_sizes = upsample_kernel_sizes  # [16,16,4,4]
        self.n_speakers = n_speakers  # 0
        self.gin_channels = gin_channels  # 0

        self.use_sdp = use_sdp  # True

        # 得到文本的先验分布
        self.enc_p = TextEncoder(n_vocab,
                                 inter_channels,
                                 hidden_channels,
                                 filter_channels,
                                 n_heads,
                                 n_layers,
                                 kernel_size,
                                 p_dropout)
        # 后验编码器
        self.enc_q = PosteriorEncoder(spec_channels,
                                      inter_channels,
                                      hidden_channels,
                                      5,
                                      1,
                                      16,
                                      gin_channels=gin_channels)
        # 对先验分布进行提高表达能力的flow
        self.flow = ResidualCouplingBlock(inter_channels,
                                          hidden_channels,
                                          5,
                                          1,
                                          4,
                                          gin_channels=gin_channels)
        self.dec = Generator(inter_channels,
                             resblock,
                             resblock_kernel_sizes,
                             resblock_dilation_sizes,
                             upsample_rates,
                             upsample_initial_channel,
                             upsample_kernel_sizes,
                             gin_channels=gin_channels)

        if self.use_sdp:
            # 随机时长（说话韵律节奏）预测器
            self.dp = StochasticDurationPredictor(hidden_channels,
                                                  192,
                                                  3,
                                                  0.5,
                                                  4,
                                                  gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(hidden_channels,
                                        256,
                                        3,
                                        0.5,
                                        gin_channels=gin_channels)

        if self.n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        """
        Args:
        x: [B, L]=[1, 199], 199是文本音素内容长度。x是文本音素
        x_lengths: [B]=[199], 存的是文本音素长度。x_lengths是文本音素长度
        y: [B,spec_channels, T]=[1, 513, 299]
        y_lengths: [B]=[1]

        Return:
        o: [B,1,Time]=【1,1,8192】
        l_length: [B]=[1]
        attn: [B,1,T,D]=[1,1,299,127]
        ids_slice: [B]=[1]
        x_mask: [B,1,D]=[1,1,127]
        y_mask: [B,1,T]=[1,1,299]
        z: [B,H,T]=[1,192,299]
        z_p: [B,H,T]=[1,192,299]
        m_p: [B,H,T]=[1,192,299]
        logs_p: [B,H,T]=[1,192,299]
        m_q: [B,H,T]=[1,192,299]
        logs_q: [B,H,T]=[1,192,299]
        """
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths)  # x是编码后，m_p 是均值，logs_p是方差,x_mask是文本mask
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)  # z的时间维度和y一致
        z_p = self.flow(z, y_mask, g=g)  # 后验中的z经过逆flow得到的z_p就是f(z,c)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) -
                                  logs_p, [1], keepdim=True)  # [b, 1, t_s]
            # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent2 = torch.matmul(-0.5 *
                                     (z_p ** 2).transpose(1, 2), s_p_sq_r)
            # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r,
                                  [1], keepdim=True)  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(
                x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(
                neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        w = attn.sum(2)  # 每个text对应频谱有多少个是这里定的。
        # 这里的l_length表示的是loss_length
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)  # l_length是对数似然
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum(
                (logw - logw_)**2, [1, 2]) / torch.sum(x_mask)  # for averaging

        # expand prior
        m_p = torch.matmul(attn.squeeze(
            1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(
            1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
        """
        Args:
        x: [B, L]=[1, 199], 199是文本音素内容长度。x是文本音素
        x_lengths: [B]=[199], 存的是文本音素长度。x_lengths是文本音素长度

        Return:
        o: [B,1,Time]=【1,1,8192】
        attn: [B,1,T,D]=[1,1,299,127]
        y_mask: [B,1,T]=[1,1,299]
        z: [B,H,T]=[1,192,299]
        z_p: [B,H,T]=[1,192,299] 
        m_p: [B,H,T]=[1,192,299]
        logs_p: [B,H,T]=[1,192,299]
        """
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths)  # x.size()=[1, 199], x_lengths.size()=[199]
        if self.n_speakers > 1:
            g = self.emb_g(sid).unsqueeze(-1)  # [B, gin_channels, 1]
        else:
            g = None

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True,
                           noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale  # [B,1,T]
        w_ceil = torch.ceil(w)
        # y_lengths.size()=[1],y_lengths changes every time
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(
            y_lengths, None), 1).to(x_mask.dtype)  # [B,1,y_lengths]=[1,1,454]
        # attn_mask.size()=[B,1,y_lengths,L]=[1,1,454,199]
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        # attm.size()=[B,1,y_lengths,L]=[1,1,454,199]
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(
            1), m_p.transpose(1, 2)).transpose(1, 2)  # m_p.size()=[B,L,y_lengths]
        logs_p = torch.matmul(attn.squeeze(
            1), logs_p.transpose(1, 2)).transpose(1, 2)  # logs_p.size()=[B,L,y_lengths]

        # z_p.size()=[B,L,y_lengths]
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        # z.size()=[B,H,y_lengths]
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)


class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels  # 513
        self.out_channels = out_channels  # 192
        self.hidden_channels = hidden_channels  # 192
        self.kernel_size = kernel_size  # 5
        self.dilation_rate = dilation_rate  # 1
        self.n_layers = n_layers  # 16
        self.gin_channels = gin_channels  # 0

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size,
                              dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        """
        Args:
        x: [B,spec_channels, T]=[1, 513, 299], linear_spectrogram
        x_lengths: [B]=[1]

        Return:
        z: [B, out_channels, T]=[1, 192, 299]
        m: [B, out_channels, T]=[1, 192, 299]
        logs: [B, out_channels, T]=[1, 192, 299]
        x_mask: [B, 1, T]=[1, 1, 299]
        """
        x_mask = torch.unsqueeze(commons.sequence_mask(
            x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)  # 加入sid作为条件
        stats = self.proj(x) * x_mask  # proj映射x到两个统计量
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask  # 采样(重参数化)得到的z
        return z, m, logs, x_mask


class TextEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.n_vocab = n_vocab  # 178
        self.out_channels = out_channels  # 192
        self.hidden_channels = hidden_channels  # 192
        self.filter_channels = filter_channels  # 768
        self.n_heads = n_heads  # 2
        self.n_layers = n_layers  # 6
        self.kernel_size = kernel_size  # 3
        self.p_dropout = p_dropout  # 0.1

        # 单词映射成embedding向量， n_vocab个单词，每个单词维度是hidden_channels
        self.emb = nn.Embedding(self.n_vocab, self.hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0,
                        hidden_channels**-0.5)  # 权重进行正态分布初始化

        self.encoder = attentions.Encoder(self.hidden_channels,
                                          self.filter_channels,
                                          self.n_heads,
                                          self.n_layers,
                                          self.kernel_size,
                                          self.p_dropout)  # 就是transformer encoder
        self.proj = nn.Conv1d(self.hidden_channels,
                              self.out_channels * 2, 1)  # 相当于MLP

    def forward(self, x, x_lengths):
        """
        Args:
        x: [B,L]=[1,199], 是音素内容
        x_lengths: [B]=[199], 是音素长度

        Return:
        x: [B,H,L]=[1,192,199], 是编码后的音频特征
        m: [B,H,L]=[1,192,199], 是均值
        logs: [B,H,L]=[1,192,199], 是logvar
        x_mask: [B,1,L]=[1,1,199], 是x的mask
        """
        x = self.emb(
            x) * math.sqrt(self.hidden_channels)  # x.size()= [b, t, hidden]， math.sqrt(self.hidden_channels)是为了缩放
        x = x.transpose(1, -1)  # x.size()=[b,hidden,t]
        x_mask = torch.unsqueeze(
            commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)  # x_mask.size()=[b,1,t]=[1,1,199]

        # after encoder x.size()=[b,hidden,t]=[1,192,199]
        # 为了确保在计算统计量（stats）时，只有有效的音素位置被考虑，而忽略填充的位置。
        # x_mask 中的值在有效位置为1，而在填充位置为0。
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask  # stats=[b,2*hidden,t]=[1,384,199]

        # m.size()=[b,hidden,t]=[1,192,199], logs.size()=[b,hidden,t]=[1,192,199], m是mean，logs是logvar
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class StochasticDurationPredictor(nn.Module):
    def __init__(self,
                 in_channels,
                 filter_channels,
                 kernel_size,
                 p_dropout,
                 n_flows=4,
                 gin_channels=0):
        super().__init__()
        filter_channels = in_channels  # 192
        self.in_channels = in_channels  # 192
        self.filter_channels = filter_channels  # 192
        self.kernel_size = kernel_size  # 3
        self.p_dropout = p_dropout  # 0.5
        self.n_flows = n_flows  # 4
        self.gin_channels = gin_channels  # 0

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(modules.ConvFlow(
                2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(modules.ConvFlow(
                2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        """
        Args:
        x: [B,H,L]=[1,192,199]， 文本编码后特征
        x_mask: [B,1,L]=[1,1,199]
        w: 时长
        g: 状态

        Return:
        logw: [B,1,L]=[1,1,199]
        """
        x = torch.detach(x)  # 分离梯度，梯度到此阶段不回传

        # After self.pre x.size()=[B,H,L]=[1,192,199]，是一个pointwise卷积
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask  # x.size()=[B,H,L]=[1,192,199]，到这里的x就是论文里的c

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask  # h_w就是公式里的d

            e_q = torch.randn(w.size(0), 2, w.size(2)).to(
                device=x.device, dtype=x.dtype) * x_mask  # 使用变分增广flow 所以是2，不是1。
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q

            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask  # 已经得到了u，v(z1)

            z0 = (w - u) * x_mask
            # 下边这两行是后验分布的对数似然。
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) +
                                      F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) +
                             (e_q**2)) * x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z**2))
                            * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(
                device=x.device, dtype=x.dtype) * noise_scale  # z.size()=[B,2,L]=[1,2,199]
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            # z0 and z1 size are all [B,1,L]=[1,1,199]
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0):
        super().__init__()
        self.channels = channels  # 192
        self.hidden_channels = hidden_channels  # 192
        self.kernel_size = kernel_size  # 5
        self.dilation_rate = dilation_rate  # 1
        self.n_layers = n_layers  # 4
        self.n_flows = n_flows  # 4
        self.gin_channels = gin_channels  # 0

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels,
                              kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))  # 耦合flow
            # 这个是颠倒上边的flow，改变上边变化的部分，否则只有固定的一半在变化。
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Args:
        x: [B,L, y_lengths]=[1,192,454]
        x_mask: [B,1,y_lengths]=[1,1,454]

        Return:
        x: [B,L, y_lengths]
        """

        if not reverse:
            for flow in self.flows:  # 正向q->p
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):  # 反向p->q
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class Generator(nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)  # 3
        self.num_upsamples = len(upsample_rates)  # 4
        self.conv_pre = nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # self.ups.append(nn.utils.weight_norm(nn.ConvTranspose1d(
            #     upsample_initial_channel // (2**i), upsample_initial_channel // (2**(i + 1))), k, u, padding=(k - u) // 2))
            self.ups.append(nn.utils.weight_norm(
                            nn.ConvTranspose1d(upsample_initial_channel // (2**i), upsample_initial_channel // (2**(i + 1)),
                                               k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(commons.init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        """
        Args:
        x: [B,DU, y_lengths]=[1,192,454]

        Return:
        x: [B,1,T]
        """
        x = self.conv_pre(x)  # [B,DU, y_lengths]
        if not g is None:
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


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm  # False
        norm_f = nn.utils.weight_norm if use_spectral_norm == False else nn.utils.spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1),
                                 padding=(get_padding(kernel_size, 1), 0))),
                norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1),
                                 padding=(get_padding(kernel_size, 1), 0))),
                norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1),
                                 padding=(get_padding(kernel_size, 1), 0))),
                norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1),
                                 padding=(get_padding(kernel_size, 1), 0))),
                norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1,
                                 padding=(get_padding(kernel_size, 1), 0))),
            ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.size()
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
        super().__init__()
        norm_f = nn.utils.weight_norm if use_spectral_norm == False else nn.utils.spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
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
