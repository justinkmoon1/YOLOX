import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Backbone, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet18 = models.resnet18(pretrained=pretrained)

        # Extract the feature extraction part (excluding the final fully connected layer)
        self.features = nn.Sequential(*list(resnet18.children())[:-2])

    def forward(self, x):
        # Forward pass through the feature extraction layers
        return self.features(x)
