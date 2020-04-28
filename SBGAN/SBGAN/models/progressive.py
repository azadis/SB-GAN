import numpy as np
import torch.nn as nn


import torch

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sbgan_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, sbgan_dir) 
from SBGAN import modules


class ProGANGenerator(nn.Module):
    def __init__(self, latent_dim=512, max_dim=128, rgb=True, num_semantics=3, T=1, aspect_ratio=1):
        super().__init__()
        if (max_dim & (max_dim - 1)) != 0 or max_dim == 0:
            raise ValueError("max_dim must be power of two")
        self.max_scale = int(np.log2(max_dim)) - 1
        self.scale = 1
        blocks = []
        for i in range(self.max_scale):
            in_dim, out_dim = self._feature_dims_for_block(i)
            blocks.append(ProGANGeneratorBlock(in_dim, out_dim, initial=i == 0, num_semantics=num_semantics, rgb=rgb, T=T, aspect_ratio=aspect_ratio))

        self.blocks = nn.ModuleList(blocks)
        self.rgb = rgb

    @property
    def max_dim(self):
        return 2 ** (self.max_scale + 1)

    @property
    def res(self):
        return 2 ** (self.scale + 1)

    @res.setter
    def res(self, val):
        if (val & (val - 1)) != 0 or val == 0:
            raise ValueError("res must be power of two")
        self.scale = int(np.log2(val)) - 1

    def _feature_dims_for_block(self, i):
        in_dim = min(512, 2 ** (13 - i))
        out_dim = min(512, 2 ** (13 - i - 1))
        return in_dim, out_dim

    def forward(self, x):
        for i in range(self.scale):
            x = self.blocks[i](x)
        if self.rgb:
            out = self.blocks[self.scale - 1].rgb(x)
        else:
            out = self.blocks[self.scale - 1].mask(x)
        return out

    def interpolate(self, x, alpha):
        for i in range(self.scale - 1):
            x = self.blocks[i](x)
        if self.rgb:
            out_coarse = self.blocks[self.scale - 2].rgb(x, double=True)
        else:
            out_coarse = self.blocks[self.scale - 2].mask(x, double=True)

        x = self.blocks[self.scale - 1](x)

        if self.rgb:
            out_fine = self.blocks[self.scale - 1].rgb(x)
        else:
            out_fine = self.blocks[self.scale - 1].mask(x)

        return (1 - alpha) * out_coarse + alpha * out_fine


class ProGANGeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, initial=False, num_semantics=3, rgb=True, T=1, aspect_ratio=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial = initial

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.norm = modules.PixelNorm()
        self.upsample = modules.NearestInterpolate(scale_factor=2)
        if initial:
            if aspect_ratio==1:
                self.h0=4
                self.w0=4
            else:
                self.h0=4
                self.w0=aspect_ratio*self.h0
            self.fc1 = modules.EqualizedLinear(
                in_channels, out_channels * self.h0 * self.w0, gain=np.sqrt(2) / 4
            )
        else:
            self.conv1 = modules.EqualizedConv2d(
                in_channels, out_channels, 3, padding=1
            )
        self.conv2 = modules.EqualizedConv2d(out_channels, out_channels, 3, padding=1)
        if rgb:
            self.to_rgb = modules.EqualizedConv2d(out_channels, 3, 1, gain=1)
        else:
            self.to_mask = modules.EqualizedConv2d(out_channels, num_semantics, 1, gain=1)
            self.softmax = nn.Softmax2d()
    def forward(self, x):
        if self.initial:
            x = self.norm(x)
            x = self.fc1(x)
            x = x.view(x.size(0), -1, self.h0, self.w0)
        else:
            x = self.upsample(x)
            x = self.conv1(x)
        x = self.norm(self.lrelu(x))
        x = self.norm(self.lrelu(self.conv2(x)))
        return x

    def rgb(self, x, double=False):
        if double:
            x = self.upsample(x)
        return self.to_rgb(x)

    def mask(self, x, double=False, num_semantics=3, T=1):
        if double:
            x = self.upsample(x)
        x = self.to_mask(x)
        # x = torch.div(x, T)
        x = self.softmax(x)
        return x

class ProGANDiscriminator(nn.Module):
    def __init__(self, max_dim=128, rgb=True, num_semantics=3, aspect_ratio=1):
        super().__init__()
        if (max_dim & (max_dim - 1)) != 0 or max_dim == 0:
            raise ValueError("max_dim must be power of two")
        self.max_scale = int(np.log2(max_dim)) - 1
        self.scale = 1
        if not isinstance(num_semantics, list):
            num_semantics = [num_semantics]*self.max_scale
        blocks = []
        for i in range(self.max_scale):
            block_i = self.max_scale - 1 - i
            in_dim, out_dim = self._feature_dims_for_block(block_i)
            blocks.append(
                ProGANDiscriminatorBlock(in_dim, out_dim, final=i == self.max_scale - 1, rgb=rgb, num_semantics=num_semantics[self.max_scale - 1 - i], aspect_ratio=aspect_ratio)
            )
        self.blocks = nn.ModuleList(blocks)
        self.rgb = rgb

    @property
    def max_dim(self):
        return 2 ** (self.max_scale + 1)

    @property
    def res(self):
        return 2 ** (self.scale + 1)

    @res.setter
    def res(self, val):
        if (val & (val - 1)) != 0 or val == 0:
            raise ValueError("dim must be power of two")
        self.scale = int(np.log2(val)) - 1

    def _feature_dims_for_block(self, i):
        in_dim = min(512, 2 ** (13 - i - 1))
        out_dim = min(512, 2 ** (13 - i))
        return in_dim, out_dim

    def forward(self, x, loss_gen=False, loss_disc=False, interpolate=False, alpha=0,
        x_real=torch.Tensor([]),
        x_fake=torch.Tensor([]),
        d_real=torch.Tensor([]),
        d_fake=torch.Tensor([])):
        if self.rgb:
            x = self.blocks[self.max_scale - self.scale].rgb(x)
        else:
            x = self.blocks[self.max_scale - self.scale].mask(x)

        for i in range(self.max_scale - self.scale, self.max_scale):
            x = self.blocks[i](x)
        if loss_gen:
            gan_loss = -x.mean()
            return x, gan_loss
        elif loss_disc:
            grad_penalty = self.gradient_penalty(x_real, x_fake, interpolate, alpha)
            return d_fake.mean() - d_real.mean() + self.lambda_ * grad_penalty
        else:
            return x

    def gradient_penalty(self, x_real, x_fake, interpolate, alpha):
        n = x_real.size(0)
        device = x_real.device

        alpha = torch.rand(n)
        alpha = alpha.to(device)
        alpha = alpha[:, None, None, None]

        interpolates = alpha * x_real.detach() + (1 - alpha) * x_fake.detach()
        interpolates.requires_grad = True
        if interpolate:
            dis_interpolates = self.interpolate(interpolates, alpha)
        else:
            dis_interpolates = self.forward(interpolates)

        grad_outputs = torch.ones_like(dis_interpolates).to(device)
        grad = torch.autograd.grad(
            outputs=dis_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad = grad.view(grad.size(0), -1)

        penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return penalty


    def interpolate(self, x, alpha):
        if self.rgb:
            x_fine = self.blocks[self.max_scale - self.scale].rgb(x)
            x_coarse = self.blocks[self.max_scale - self.scale + 1].rgb(x, half=True)
        else:
            x_fine = self.blocks[self.max_scale - self.scale].mask(x)
            x_coarse = self.blocks[self.max_scale - self.scale + 1].mask(x, half=True)

        x_fine = self.blocks[self.max_scale - self.scale](x_fine)
        x = (1 - alpha) * x_coarse + alpha * x_fine
        for i in range(self.max_scale - self.scale + 1, self.max_scale):
            x = self.blocks[i](x)
        return x


class ProGANDiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, final=False,rgb=True, num_semantics=3, aspect_ratio=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final = final

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.downsample = nn.AvgPool2d(2)
        if rgb:
            self.from_rgb = modules.EqualizedConv2d(3, in_channels, 1, gain=1)
        else:
            self.from_mask = modules.EqualizedConv2d(num_semantics, in_channels, 1, gain=1)
        if final:
            ar=aspect_ratio
        else:
            ar=1

        self.conv1 = modules.EqualizedConv2d(in_channels, in_channels, 3, stride=(1, ar), padding=1)
        if final:
            self.conv2 = modules.EqualizedConv2d(in_channels, out_channels, 4)
            self.fc3 = modules.EqualizedLinear(out_channels, 1)
        else:
            self.conv2 = modules.EqualizedConv2d(
                in_channels, out_channels, 3, padding=1
            )

    def forward(self, x):

        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        if self.final:
            x = x.view(x.size(0), -1)
            x = self.fc3(x)
        else:
            x = self.downsample(x)
        return x

    def rgb(self, x, half=False):
        if half:
            x = self.downsample(x)
        return self.lrelu(self.from_rgb(x))


    def mask(self, x, half=False):
        if half:
            x = self.downsample(x)
        return self.lrelu(self.from_mask(x))

