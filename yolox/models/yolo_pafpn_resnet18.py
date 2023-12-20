import torch.nn as nn
import torch.nn.functional as F
from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck

class YOLOPAFPN_ResNet18(nn.Module):
    def __init__(self, in_channels=(64, 64, 128, 256, 512), out_channels=256):
        super(YOLOPAFPN_ResNet18, self).__init__()

        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channel, out_channels, kernel_size=1, stride=1, padding=0)
            for in_channel in reversed(in_channels)
        ])

        # Top-Down path
        self.top_down_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            for _ in range(len(in_channels) - 1)
        ])

        # Bottom-Up path
        self.bottom_up_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            for _ in range(len(in_channels) - 1)
        ])

    def forward(self, x):
        lateral_features = [lateral_conv(x[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # Top-Down path
        top_down_features = [lateral_features[-1]]
        for i in range(len(lateral_features) - 2, -1, -1):
            top_down_features.insert(0, self.top_down_convs[i](F.interpolate(top_down_features[0], scale_factor=2, mode='nearest')))

        # Bottom-Up path
        bottom_up_features = [lateral_features[0]]
        for i in range(1, len(lateral_features) - 1):
            bottom_up_features.append(self.bottom_up_convs[i-1](F.interpolate(bottom_up_features[-1], scale_factor=0.5, mode='nearest')))

        # Aggregation
        pafpn_features = [lateral + top_down + bottom_up for lateral, top_down, bottom_up in zip(lateral_features, top_down_features, bottom_up_features)]

        return pafpn_features
