import os
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class MNISTDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.ids = []
        self._load_dataset()

    def _load_dataset(self):
        if self.train:
            folder_path = os.path.join(self.root_dir, 'mnist_train')
            label_path = os.path.join(self.root_dir, 'mnist_train_labels.txt')
        else:
            folder_path = os.path.join(self.root_dir, 'mnist_test')
            label_path = os.path.join(self.root_dir, 'mnist_test_labels.txt')
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            self.image_paths.append(file_path)
        with open(label_path, 'r') as f:
            line = f.readline()
            while line:
                if line != '':
                    label_str = line.strip()
                    self.labels.append(int(label_str.split()[1]))
                line = f.readline()
        self.ids = list(range(len(self.image_paths)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self._load_image(idx)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        id_value = self.ids[idx]
        return image, label, id_value

    def _load_image(self, index):
        return Image.open(self.image_paths[index])
