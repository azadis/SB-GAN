import torch.nn as nn
import torch.nn.functional as F


class NearestInterpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


class BilinearInterpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=True,
        )
