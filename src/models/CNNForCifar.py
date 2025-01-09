import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim


class CNNForCifar(nn.Module):
    def __init__(self, color, feature_size):
        super(CNNForCifar, self).__init__()
        if color:
            img_channels = 3
        else:
            img_channels = 1
        self.conv1 = nn.Conv2d(img_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Linear(64 * 4 * 4, feature_size)
        # self.relu4 = nn.PReLU()
        # self.fc2 = nn.Linear(128, feature_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)

        return x

    def l2_regularization_loss(self):
        reg_loss = 0
        for param in self.parameters():
            reg_loss += torch.norm(param, 2) ** 2
        return reg_loss
