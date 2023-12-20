import torch.nn as nn
from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck

class ResNet101(nn.Module):
    def __init__(self, in_channels=3, out_features=("res3", "res4", "res5")):
        super(ResNet101, self).__init__()
        assert out_features, "please provide output features of ResNet101"
        self.out_features = out_features

        # Initial Convolutional Layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet Blocks
        self.res3 = self._make_resnet_layer(64, 256, 3, stride=1)
        self.res4 = self._make_resnet_layer(256, 512, 4, stride=2)
        self.res5 = self._make_resnet_layer(512, 1024, 23, stride=2)

    def _make_resnet_layer(self, in_channels, out_channels, blocks, stride):
        layers = [
            Bottleneck(in_channels, out_channels, stride)
        ]
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["res2"] = x  # Assuming the output after the initial convolution is referred to as "res2"
        x = self.res3(x)
        outputs["res3"] = x
        x = self.res4(x)
        outputs["res4"] = x
        x = self.res5(x)
        outputs["res5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Assume you have the necessary BaseConv, CSPLayer, DWConv, Focus, and SPPBottleneck classes already defined in your "network_blocks" module.
