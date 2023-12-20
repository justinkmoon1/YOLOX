import torch.nn as nn
import torch.nn.functional as F
from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck

class MobileNetV3(nn.Module):
    def __init__(self, in_channels=3, out_features=("mob3", "mob4", "mob5"), depth=1.0, width=1.0):
        super(MobileNetV3, self).__init__()
        assert out_features, "please provide output features of MobileNetV3"
        self.out_features = out_features
        self.depth = depth
        self.width = width

        # Initial Convolutional Layer
        initial_channels = int(16 * width)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.Hardswish(inplace=True),
        )

        # MobileNetV3 Blocks
        self.mob3 = self._make_mobilenet_layer(initial_channels, 24, 3, stride=2, exp_ratio=3, se_ratio=0.25)
        self.mob4 = self._make_mobilenet_layer(24, 40, 4, stride=2, exp_ratio=3, se_ratio=0.25)
        self.mob5 = self._make_mobilenet_layer(40, 80, 6, stride=2, exp_ratio=3, se_ratio=0.25)

    def _make_mobilenet_layer(self, in_channels, out_channels, blocks, stride, exp_ratio, se_ratio):
        layers = [
            MobileNetV3Block(in_channels, out_channels, stride, exp_ratio, se_ratio, width=self.width)
        ]
        for _ in range(1, blocks):
            layers.append(MobileNetV3Block(out_channels, out_channels, 1, exp_ratio, se_ratio, width=self.width))
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["mob2"] = x
        x = self.mob3(x)
        outputs["mob3"] = x
        x = self.mob4(x)
        outputs["mob4"] = x
        x = self.mob5(x)
        outputs["mob5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

class MobileNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, exp_ratio, se_ratio, width=1.0):
        super(MobileNetV3Block, self).__init__()

        # Expansion layer
        exp_channels = int(in_channels * exp_ratio * width)
        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels, exp_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp_channels),
            nn.Hardswish(inplace=True),
        )

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(exp_channels, exp_channels, kernel_size=3, stride=stride, padding=1, groups=exp_channels, bias=False),
            nn.BatchNorm2d(exp_channels),
            nn.Hardswish(inplace=True),
        )

        # Squeeze-and-Excitation (SE) block
        se_channels = int(in_channels * se_ratio * width)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(exp_channels, se_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Hardswish(inplace=True),
            nn.Conv2d(se_channels, exp_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        # Linear projection
        self.linear = nn.Sequential(
            nn.Conv2d(exp_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Identity skip connection (if stride is 1 and input and output channels are the same)
        self.has_identity = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        identity = x

        x = self.expansion(x)
        x = self.depthwise(x)

        # Squeeze-and-Excitation
        w = self.se(x)
        x = x * w

        x = self.linear(x)

        # Skip connection
        if self.has_identity:
            x = x + identity

        return x


# Assume you have the necessary YOLOX blocks and layers defined in your code.
