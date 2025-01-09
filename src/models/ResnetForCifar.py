import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class ResnetForCifar(nn.Module):
    def __init__(self, color, feature_size):
        super(ResnetForCifar, self).__init__()

        if color:
            img_channels = 3
        else:
            img_channels = 1
        self.conv1 = nn.Conv2d(img_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = _make_layer(16, 16, num_blocks=2)
        self.layer2 = _make_layer(16, 32, num_blocks=2, stride=2)

        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(512, feature_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def _make_layer(in_channels, out_channels, num_blocks, stride=1):
    layers = [ResidualBlock(in_channels, out_channels, stride)]
    for i in range(1, num_blocks):
        layers.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
