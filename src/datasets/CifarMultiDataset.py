import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class CifarMultiDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, index):
        data1, label1, id1 = self.dataset1[index]
        data2, label2, id2 = self.dataset2[index]
        return data1, data2, label1, id1
