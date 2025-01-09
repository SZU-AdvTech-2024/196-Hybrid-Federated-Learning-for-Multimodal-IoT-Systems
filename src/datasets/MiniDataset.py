import os
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class MiniDataset(Dataset):
    def __init__(self, dataset, mini_dataset_ids):
        self.dataset = dataset
        self.mini_dataset_ids = mini_dataset_ids

    def __len__(self):
        return len(self.mini_dataset_ids)

    def __getitem__(self, idx):
        color, gray, label, id_value = self.dataset[self.mini_dataset_ids[idx]]
        return color, gray, label, id_value

