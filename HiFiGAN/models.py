import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (init_weights, get_padding)

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.h = h
        # channels=256
        self.convs1 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                 dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                 dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                 dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]))),
        )
        self.convs1.apply(init_weights)  # self.convs1 has 3 conv1d layers

        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                           padding=get_padding(kernel_size, 1))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                           padding=get_padding(kernel_size, 1))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                           padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)  # self.convs2 has 3 conv1d layers

    def forward(self, x):
        """
        Args:
        x: [B,D,F]

        Return:
        x: [B,D,F]
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            nn.utils.remove_weight_norm(l)
        for l in self.convs2:
            nn.utils.remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.h = h
        self.convs = nn.Sequential(
            nn.utils.weight_norm(nn.utils.weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0])))),
            nn.utils.weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1])))
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for l in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = l(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            nn.utils.remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super().__init__()

        self.h = h
        # h.resblock_kernel_sizes=[3,7,11]
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)  # [8,8,2,2]
        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))  # upsample_initial_channel=512
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        # upsample_kernel_sizes=[16,16,4,4]
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.utils.weight_norm(
                nn.ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                   h.upsample_initial_channel // (2**(i + 1)),
                                   k, u, padding=(k - u) // 2)))
        # self.ups has 4 convtranspose1d layers

        self.resblocks = nn.ModuleList()
        # h.resblock_dilation_sizes=[(1,3,5),(1,3,5),(1,3,5),(1,3,5)]
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2**(i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                # self.resblocks has 4*3=12 resblock layers
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = nn.utils.weight_norm(
            nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        """
        Args:
        x: [B,D,F]=[16,80,32], 80 dim mel-spec, 32 frames with hop_size=256

        Return:
        x: [B,1,T]=[16,1,8192]
        """
        x = self.conv_pre(x)  # [B, 512, 32]
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None

            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)  # B,D,DT（time dim of output, not frame or samples）
        x = self.conv_post(x)  # B,1,T
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            nn.utils.remove_weight_norm(l)  # ConvTranspose
        for l in self.resblocks:
            l.remove_weight_norm()  # Resblock
        nn.utils.remove_weight_norm(self.conv_pre)
        nn.utils.remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.Sequential(
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1),
                   (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1),
                   (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1),
                   (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1),
                   (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        )
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        """
        Args:
        x: [B,1,segments]=[16,1,8192], segments=8192

        Return:
        x: [B,1,T,2]
        fmap: len(fmap)=len(convs)+1(conv_post), each element is [B,1,T,C] T,C are changed in each layer
        """
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
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.discriminators = nn.Sequential(
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        )

    def forward(self, y, y_hat):
        """
        Args:
        y: [B,1,segments]=[16,1,8192], segments=8192
        y_hat: [B,1,segments]=[16,1,8192], segments=8192

        Return:
        y_d_rs: len=5, each element is [B,1,T,C]
        y_d_gs: len=5, each element is [B,1,T,C]
        fmap_rs: len=5 feature maps, list of list of 4d tensor
        fmap_gs: len=5 feature maps, list of list of 4d tensor
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.Sequential(
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, padding=20, groups=4)),
            norm_f(nn.Conv1d(128, 256, 41, 2, padding=20, groups=16)),
            norm_f(nn.Conv1d(256, 512, 41, 4, padding=20, groups=16)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, padding=20, groups=16)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, padding=20, groups=16)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2))
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        """
        Args:
        x: [B,1,segments]=[16,1,8192], segments=8192

        Return:
        x: [B,1,T]
        fmap: len(fmap)=len(convs)+1(conv_post), each element is [B,D,T] D,T are changed in each layer
        """
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminators = nn.Sequential(
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS()
        )
        self.meanpools = nn.Sequential(
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        )

    def forward(self, y, y_hat):
        """
        Args:
        y: [B,1,segments]=[16,1,8192], segments=8192
        y_hat: [B,1,segments]=[16,1,8192], segments=8192

        Return:
        y_d_rs: len=3, each element is [B,1,C]
        y_d_gs: len=3, each element is [B,1,C]
        fmap_rs: len=3 feature maps, list of list of 3d tensor
        fmap_gs: len=3 feature maps, list of list of 3d tensor
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    """
    Args:
    fmap_r: list of list of 3/4d tensor
    fmap_g: list of list of 3/4d tensor

    Return:
    loss: scaler
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Args:
    disc_real_outputs: len=5
    disc_generated_outputs: len=5

    Return:
    loss: scaler
    r_losses: len=5, list of scaler
    g_losses: len=5, list of scaler
    """
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """
    Args:
    disc_outputs: list

    Return:
    loss: scaler
    gen_losses: list of scaler
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg)**2)  # scaler
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
