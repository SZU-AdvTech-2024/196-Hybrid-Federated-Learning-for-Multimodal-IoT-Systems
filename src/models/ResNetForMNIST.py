import torch
import torch.nn as nn
import torchvision.models as models


# class ResNetForMNIST(nn.Module):
#
#     def __init__(self, color, feature_size):
#         super(ResNetForMNIST, self).__init__()
#         self.feature_size = feature_size
#         if color:
#             in_channels = 3
#         else:
#             in_channels = 1
#         self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self._make_layer(16, 16, 2)
#         self.layer2 = self._make_layer(16, 32, 2, stride=2)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(32, self.feature_size)
#         self.l2_reg = nn.MSELoss()
#
#     def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
#         layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
#                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
#         for _ in range(1, num_blocks):
#             layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
#             layers.append(nn.BatchNorm2d(out_channels))
#             layers.append(nn.ReLU(inplace=True))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)
#
#     def l2_regularization_loss(self):
#         reg_loss = 0
#         for param in self.parameters():
#             reg_loss += torch.norm(param, 2) ** 2
#         return reg_loss


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNetForMNIST(nn.Module):
    def __init__(self, color, feature_size):
        super(ResNetForMNIST, self).__init__()
        self.in_channels = 64

        if color:
            self.image_channels = 3
        else:
            self.image_channels = 1
        self.conv1 = nn.Conv2d(self.image_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, feature_size)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def l2_regularization_loss(self):
        reg_loss = 0
        for param in self.parameters():
            reg_loss += torch.norm(param, 2) ** 2
        return reg_loss
