import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CifarDataset(Dataset):
    def __init__(self, root, train=True, download=True, transform=None):
        self.cifar10_dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=download, transform=transform)
        self.sample_ids = list(range(len(self.cifar10_dataset)))

    def __len__(self):
        return len(self.cifar10_dataset)

    def __getitem__(self, index):
        image, label = self.cifar10_dataset[index]
        id_value = self.sample_ids[index]
        return image, label, id_value
