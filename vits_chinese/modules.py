import torch
import math
import torch.nn as nn
import torch.nn.functional as F

import commons
from commons import (get_padding, init_weights)
from transforms import piecewise_rational_quadratic_transform

LRELU_SLOPE = 0.1


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        """
        Args:
        x: [B,H,L]=[1,192,199]

        Return:
        x: [B,H,L]=[1,192,199]
        """
        x = x.transpose(1, -1)  # [B,L,H]=[1,199,192]
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        """
        Args:
        x: [B,H,L]=[1,192,199]
        x_mask

        Return:
        x
        y
        logdet
        """
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        """
        Args:
        x: [B,1,L]=[1,1,199]
        x_mask: [B,1,L]=[1,1,199]

        Return:
        x: [B,2,L]=[1,2,199]
        y
        logdet
        """
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        """
        Args:
        x: [B,2,L]=[1,2,199]

        Return:
        x: [B,2,L]=[1,2,199]
        logdet
        """
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(device=x.device, dtype=x.dtype)
            return x, logdet
        else:
            return x


class ResidualCouplingLayer(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 p_dropout=0,
                 gin_channels=0,
                 mean_only=False):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate,
                      n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = nn.Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Args:
        x: [B,2,L]=[1,2,199]
        x_mask

        Return:
        x:
        logdet
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class WN(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert (kernel_size % 2 == 1)
        self.hidden_channels = hidden_channels  # 192
        self.kernel_size = kernel_size  # 5
        self.dilation_rate = dilation_rate  # 1
        self.n_layers = n_layers  # 4
        self.gin_channels = gin_channels  # 0
        self.p_dropout = p_dropout  # 0

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = nn.utils.weight_norm(
                cond_layer, name='weight')

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                                 dilation=dilation, padding=padding)
            in_layer = nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(
                hidden_channels, res_skip_channels, 1)
            res_skip_layer = nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        """
        Args:
        x: [B,H,y_lengths]
        x_mask: [B,1,y_lengths]

        Return:
        output * x_mask: [B,H,y_lengths]
        """
        output = torch.zeros_like(x)  # [B,H,y_lengths]
        n_channels_tensor = torch.IntTensor(
            [self.hidden_channels])  # size()=[1], value=[192]

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)  # x_in.size()=[N,2H,y_lengths]
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset +
                        2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(
                x_in,
                g_l,
                n_channels_tensor)  # acts.size()=[B,H,y_lengths]
            acts = self.drop(acts)

            # res_skip_acts.size()=[B,2H,y_lengths]]
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)


class ResidualCouplingLayer(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 p_dropout=0,
                 gin_channels=0,
                 mean_only=False):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels  # 192
        self.hidden_channels = hidden_channels  # 192
        self.kernel_size = kernel_size  # 5
        self.dilation_rate = dilation_rate  # 1
        self.n_layers = n_layers  # 4
        self.half_channels = channels // 2  # 96
        self.mean_only = mean_only  # True

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate,
                      n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = nn.Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Args:
        x: [B,H y_lengths]=[1,192,454]
        x_mask: [B,1,y_lengths]=[1,1,454]

        Return:
        x: [B,H,y_lengths]
        logdet: 
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2,
                             1)  # x0 and x1 are all size [B,96,y_lengths]=[1,96,454]
        h = self.pre(x0) * x_mask  # [B,H,y_lengths]
        h = self.enc(h, x_mask, g=g)  # [B,H,y_lengths]
        stats = self.post(h) * x_mask  # [B,H/2, y_lengths]
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x  # [B,H,y_lengths]


class ConvFlow(nn.Module):
    def __init__(self,
                 in_channels,
                 filter_channels,
                 kernel_size,
                 n_layers,
                 num_bins=10,
                 tail_bound=5.0):
        super().__init__()
        self.in_channels = in_channels  # 2
        self.filter_channels = filter_channels  # 192
        self.kernel_size = kernel_size  # 3
        self.n_layers = n_layers  # 3
        self.num_bins = num_bins  # 10
        self.tail_bound = tail_bound  # 5.0
        self.half_channels = in_channels // 2  # 1

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size,
                             n_layers, p_dropout=0.)
        self.proj = nn.Conv1d(
            filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Args:
        x: [B,H,L]=[1,192,199] or [1,2,199]
        x_mask: [B,1,T]=[1,1,199]

        Return:
        x: [B,2,L]=[1,2,199]
        logdet: [1]
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2,
                             1)  # x0 and x1 are all size[B,1,T]=[1,1,199]
        h = self.pre(x0)  # h.size()=[B,H,L]=[1,192,199]
        h = self.convs(h, x_mask, g=g)  # h.size()=[B,H,L]=[1,192,199]
        h = self.proj(h) * x_mask  # h.size()=[B,29,L]=[1,29,199]

        b, c, t = x0.shape
        # [b, cx?, t] -> [b, c, t, ?]
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # h.size()=[1,1,199,29]

        unnormalized_widths = h[..., :self.num_bins] / \
            math.sqrt(
                self.filter_channels)  # unnormalized_widths.size()=[1,1,199,10]
        unnormalized_heights = h[..., self.num_bins:2 *
                                 self.num_bins] / math.sqrt(self.filter_channels)  # unnormalized_heights.size()=[1,1,199,10]
        # unnormalized_derivatives.size()=[1,1,199,9]
        unnormalized_derivatives = h[..., 2 * self.num_bins:]

        x1, logabsdet = piecewise_rational_quadratic_transform(x1,
                                                               unnormalized_widths,
                                                               unnormalized_heights,
                                                               unnormalized_derivatives,
                                                               inverse=reverse,
                                                               tails="linear",
                                                               tail_bound=self.tail_bound
                                                               )  # x1, logabsdet are all [B,1,L]=[1,1,199]
        x = torch.cat([x0, x1], 1) * x_mask  # [B,2,L]=[1,2,199]
        logdet = torch.sum(logabsdet * x_mask, [1, 2])  # logdet.size()=[1]
        if not reverse:
            return x, logdet
        else:
            return x


class DDSConv(nn.Module):
    """
    Dialted and Depth-Separable Convolution
    """

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.):
        super().__init__()
        self.channels = channels  # 192
        self.kernel_size = kernel_size  # 3
        self.n_layers = n_layers  # 3
        self.p_dropout = p_dropout  # 0.0

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(nn.Conv1d(
                channels, channels, kernel_size,
                groups=channels, dilation=dilation, padding=padding))
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        """
        Args:
        x: [B,H,L]=[1,192,199]
        x_mask: [B,1,T]=[1,1,199]

        Return:
        x*x_mask: [B,H,L]=[1,192,199]
        """
        if not g is None:
            x = x + g

        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)  # y.size=[B,H,L]=[1,192,199]
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                           padding=get_padding(kernel_size, dilation[0]))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                                           padding=get_padding(kernel_size, dilation[1]))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                                           padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                           padding=get_padding(kernel_size, 1))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                           padding=get_padding(kernel_size, 1))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                           padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            nn.utils.remove_weight_norm(l)
        for l in self.convs2:
            nn.utils.remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                           padding=get_padding(kernel_size, dilation[0]))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                                           padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            nn.utils.remove_weight_norm(l)
