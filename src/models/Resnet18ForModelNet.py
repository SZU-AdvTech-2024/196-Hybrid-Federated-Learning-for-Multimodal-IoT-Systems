import torch
import torch.nn as nn
import torchvision.models as models


class Resnet18ForModelNet(nn.Module):
    def __init__(self):
        super(Resnet18ForModelNet, self).__init__()
        # 加载预训练的 ResNet18 模型，并去除最后的全连接层
        self.resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # 添加一个自适应平均池化层，将特征图转换为 256 长度的特征向量
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 256)

    def forward(self, x):
        x = self.resnet(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
