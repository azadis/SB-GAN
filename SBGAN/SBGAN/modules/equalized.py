import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualizedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        gain=math.sqrt(2),
        bias=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        fan_in = nn.init._calculate_correct_fan(self.weight, "fan_in")
        std = gain / math.sqrt(fan_in)
        self.scale = std

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        w = self.scale * self.weight
        return F.conv2d(
            input, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class EqualizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, gain=math.sqrt(2), bias=True):
        super().__init__(in_features, out_features, bias=bias)
        fan_in = nn.init._calculate_correct_fan(self.weight, "fan_in")
        std = gain / math.sqrt(fan_in)
        self.scale = std

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        w = self.scale * self.weight
        return F.linear(input, w, self.bias)
