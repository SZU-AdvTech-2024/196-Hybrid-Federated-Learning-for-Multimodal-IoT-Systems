import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.datasets.ModelNetDataset import *
from src.datasets.MiniDataset import *
from src.datasets.MNISTDataset import *
from src.datasets.MNISTMDataset import *
from src.datasets.CifarDataset import *
from src.datasets.CifarMultiDataset import *


def min_max_normalize(image):
    min_val = image.min()
    max_val = image.max()
    return (image - min_val)/(max_val - min_val)


def convert_to_grayscale(image):
    # return torchvision.transforms.functional.to_grayscale(image, num_output_channels=1)
    return torchvision.transforms.Grayscale(1)(image)


def generate_dataset(dataset_type, data_path=None):
    if dataset_type == 'MNIST':
        # 定义数据预处理转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: min_max_normalize(x))
        ])
        # 加载 MNIST 训练集
        # train_dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
        train_dataset = MNISTDataset(data_path+'/mnist', train=True, transform=transform)
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # 加载 MNIST 测试集
        # test_dataset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
        test_dataset = MNISTDataset(data_path+'/mnist', train=False, transform=transform)
        # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataset, test_dataset

    elif dataset_type == 'MNIST-M':
        # 定义数据预处理转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: min_max_normalize(x))
        ])
        # 加载MNIST-M训练集
        # train_dataset = torchvision.datasets.MNISTM(root='../data', train=True, download=True, transform=transform)
        train_dataset = MNISTMDataset(data_path+'/mnist_m', train=True, transform=transform)
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # 加载MNIST-M测试集
        # test_dataset = torchvision.datasets.MNISTM(root='../data', train=False, download=True, transform=transform)
        test_dataset = MNISTMDataset(data_path+'/mnist_m', train=False, transform=transform)
        # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataset, test_dataset

    elif dataset_type == 'ModelNet10':
        # 定义数据预处理转换
        transform = transforms.Compose([
            transforms.Lambda(convert_four_to_three_channels),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 加载ModelNet训练集
        train_dataset = ModelNetDataset(root_dir=data_path, train=True, transform=transform)
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # 加载ModelNet测试集
        test_dataset = ModelNetDataset(root_dir=data_path, train=False, transform=transform)
        # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataset, test_dataset

    elif dataset_type == 'Cifar':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: min_max_normalize(x))
        ])
        train_dataset = CifarDataset(root='../data', train=True, transform=transform)
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = CifarDataset(root='../data', train=False, transform=transform)
        # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataset, test_dataset

    elif dataset_type == 'Cifar-gray':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: convert_to_grayscale(x)),
            transforms.Lambda(lambda x: min_max_normalize(x))
        ])
        train_dataset = CifarDataset(root='../data', train=True, transform=transform)
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = CifarDataset(root='../data', train=False, transform=transform)
        # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataset, test_dataset

    elif dataset_type == 'Multiple':
        transform_color = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: min_max_normalize(x))
        ])
        transform_gray = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: convert_to_grayscale(x)),
            transforms.Lambda(lambda x: min_max_normalize(x))
        ])
        train_dataset_color = CifarDataset(root='../data', train=True, transform=transform_color)
        test_dataset_color = CifarDataset(root='../data', train=False, transform=transform_color)
        train_dataset_gray = CifarDataset(root='../data', train=True, transform=transform_gray)
        test_dataset_gray = CifarDataset(root='../data', train=False, transform=transform_gray)

        train_dataset = CifarMultiDataset(train_dataset_color, train_dataset_gray)
        test_dataset = CifarMultiDataset(test_dataset_color, test_dataset_gray)
        return train_dataset, test_dataset


def generate_mini_dataloader(dataset, mini_dataset_batch_size, mini_dataset_ids):
    """
    根据全体数据集dataloader及选择的id列表生成新的dataloader
    :param dataset:
    :param mini_dataset_batch_size:
    :param mini_dataset_ids:
    :return:
    """
    mini_dataset = MiniDataset(dataset, mini_dataset_ids)
    mini_dataloader = DataLoader(mini_dataset, batch_size=mini_dataset_batch_size, shuffle=True)
    return mini_dataloader


def convert_four_to_three_channels(img):
    img_array = np.array(img)
    if img_array.shape[2] == 4:
        return Image.fromarray(img_array[:, :, :3])
    else:
        return img
