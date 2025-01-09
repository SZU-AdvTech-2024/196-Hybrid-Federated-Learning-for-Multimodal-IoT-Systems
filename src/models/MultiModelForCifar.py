import torch
import torch.nn as nn

from src.models.CNNForCifar import *
from src.models.ResnetForCifar import *
from src.models.ClassfierModel import *


class MultiModelForCifar(nn.Module):
    def __init__(self, device):
        super(MultiModelForCifar, self).__init__()
        modality_num = 2
        feature_size = 128
        class_num = 100

        self.device = device

        self.classifier = ClassifierModel(feature_size*modality_num, class_num)
        # self.color_model = CNNForCifar(True, feature_size)
        # self.gray_model = CNNForCifar(False, feature_size)
        self.color_model = ResnetForCifar(True, feature_size)
        self.gray_model = ResnetForCifar(False, feature_size)
        self.relu = nn.ReLU()

    def forward(self, color=None, gray=None):
        if color is not None:
            color = self.color_model(color)
        else:
            color = torch.zeros(*gray.shape).to(self.device)
        if gray is not None:
            gray = self.gray_model(gray)
        else:
            gray = torch.zeros(*color.shape).to(self.device)

        x = torch.cat((color, gray), dim=1)
        x = self.relu(x)
        x = self.classifier(x)

        return x
