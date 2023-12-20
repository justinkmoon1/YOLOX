import torch
import torch.nn as nn
from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck

class ResLayer(nn.Module):
    def __init__(self, channels):
        super(ResLayer, self).__init__()
        self.block1 = BaseConv(channels, channels // 2, 1, stride=1, act="lrelu")
        self.block2 = BaseConv(channels // 2, channels, 3, stride=1, act="lrelu")

    def forward(self, x):
        return x + self.block2(self.block1(x))

class Darknet53(nn.Module):
    def __init__(self, in_channels=3, out_features=("dark3", "dark4", "dark5")):
        super(Darknet53, self).__init__()
        assert out_features, "please provide output features of Darknet53"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, 32, 3, stride=1, act="lrelu"),
            *self.make_group_layer(32, num_blocks=1, stride=2)
        )
        in_channels = 64  # 32 * 2

        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks=1, stride=2),
            ResLayer(in_channels),
        )
        in_channels *= 2  # 64

        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks=2, stride=2),
            ResLayer(in_channels),
        )
        in_channels *= 2  # 128

        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks=8, stride=2),
            ResLayer(in_channels),
        )
        in_channels *= 2  # 256

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks=8, stride=2),
            ResLayer(in_channels),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

# BaseConv, CSPLayer, DWConv, Focus, SPPBottleneck classes are assumed to be defined elsewhere in the code, as they are used in the provided Darknet and CSPDarknet implementations.
