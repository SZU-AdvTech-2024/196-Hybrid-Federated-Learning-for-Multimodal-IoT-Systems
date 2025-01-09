import sys

import torch

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import argparse
import datetime

from src.datasets.DataloaderGenerator import *

# dataset_root_path = 'D:\.download\MNIST-M\data\mnist_m'
# dataset_root_path = '/home/data2/duwenfeng/datasets/MNIST'
#
# mnist_m_train_dataloader, mnist_m_test_dataloader = generate_dataloader('MNIST-M', 64, dataset_root_path,
#                                                                         load_type=True)
# mnist_train_dataloader, mnist_test_dataloader = generate_dataloader('MNIST', 64, dataset_root_path, load_type=True)


def get_color_mean_std(dataloader):
    num_channels = 3
    channel_sum = torch.zeros(num_channels)
    channel_sum_squared = torch.zeros(num_channels)
    num_images = 0
    for data in dataloader:
        images, _, _ = data
        batch_size = images.size(0)
        num_images += batch_size
        for channel in range(num_channels):
            channel_sum[channel] += images[:, channel, :, :].sum()
            channel_sum_squared[channel] += (images[:, channel, :, :] ** 2).sum()
    mean = channel_sum / (num_images * 32 * 32)
    std = torch.sqrt((channel_sum_squared / (num_images * 32 * 32)) - mean ** 2)
    return mean, std


def get_bw_mean_std(dataloader):
    pixel_sum = 0
    pixel_sum_squared = 0
    num_images = 0
    for data in dataloader:
        images, _, _ = data
        batch_size = images.size(0)
        num_images += batch_size
        pixel_sum += images.sum().item()
        pixel_sum_squared += (images ** 2).sum().item()
    mean = pixel_sum / (num_images * 32 * 32)
    std = np.sqrt(pixel_sum_squared / (num_images * 32 * 32) - mean ** 2)
    return mean, std


def find_color_neg_pixel(dataloader):
    for batch_idx, (images, _, _) in enumerate(dataloader):
        # 检查每个通道的像素值是否有负数
        for channel in range(images.shape[1]):
            negative_pixel_mask = images[:, channel, :, :] < 0
            if negative_pixel_mask.sum() > 0:
                print(f"在第{batch_idx}批数据的第{channel}通道中发现负数像素")
                return
    print("未发现负数像素")


def find_bw_neg_pixel(dataloader):
    for batch_idx, (images, _, _) in enumerate(dataloader):
        negative_pixel_mask = images < 0
        if negative_pixel_mask.sum() > 0:
            print(f"在第{batch_idx}批数据中发现负数像素")
            return
    print("未发现负数像素")


# print('mnist-m train')
# find_color_neg_pixel(mnist_m_train_dataloader)
# print('mnist-m test')
# find_color_neg_pixel(mnist_m_test_dataloader)
# print('mnist train')
# find_bw_neg_pixel(mnist_train_dataloader)
# print('mnist test')
# find_bw_neg_pixel(mnist_test_dataloader)

# mnist_m_train_mean, mnist_m_train_std = get_color_mean_std(mnist_m_train_dataloader)
# print(f'mnist-m train mean: {mnist_m_train_mean}, std: {mnist_m_train_std}')
# # mnist-m train mean: tensor([0.4579, 0.4621, 0.4082]), std: tensor([0.2519, 0.2368, 0.2587])
#
# mnist_m_test_mean, mnist_m_test_std = get_color_mean_std(mnist_m_test_dataloader)
# print(f'mnist-m test mean: {mnist_m_test_mean}, std: {mnist_m_test_std}')
# # mnist-m test mean: tensor([0.4609, 0.4644, 0.4109]), std: tensor([0.2526, 0.2379, 0.2587])
#
# mnist_train_mean, mnist_train_std = get_bw_mean_std(mnist_train_dataloader)
# print(f'mnist train mean: {mnist_train_mean}, std: {mnist_train_std}')
# # mnist train mean: 0.4547037238199928, std: 0.2337940737893796
#
# mnist_test_mean, mnist_test_std = get_bw_mean_std(mnist_test_dataloader)
# print(f'mnist test mean: {mnist_test_mean}, std: {mnist_test_std}')
# # mnist test mean: 0.45724917173306157, std: 0.23453493441072595

def min_max_normalize(image):
    min_val = image.min()
    max_val = image.max()
    return (image - min_val)/(max_val - min_val)


def convert_to_grayscale(image):
    # return torchvision.transforms.functional.to_grayscale(image, num_output_channels=1)
    return torchvision.transforms.Grayscale(1)(image)


color_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: min_max_normalize(x))
])
color_train_dataset = CifarDataset(root='../data', train=True, transform=color_transform)
color_test_dataset = CifarDataset(root='../data', train=False, transform=color_transform)

gray_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: convert_to_grayscale(x)),
            transforms.Lambda(lambda x: min_max_normalize(x))
        ])
gray_train_dataset = CifarDataset(root='../data', train=True, transform=gray_transform)
gray_test_dataset = CifarDataset(root='../data', train=False, transform=gray_transform)

print(f'color train size: {len(color_train_dataset)}, test size: {len(color_test_dataset)}')
print(f'gray train size: {len(gray_train_dataset)}, test size: {len(gray_test_dataset)}')

for idx in range(len(color_train_dataset)):
    _, color_label, color_id = color_train_dataset[idx]
    _, gray_label, gray_id = gray_train_dataset[idx]
    if color_label != gray_label or color_id != gray_id:
        print(f'train dataset is not same: color id: {color_id}, gray id: {gray_id}, color label: {color_label}, gray label: {gray_label}')

for idx in range(len(color_test_dataset)):
    _, color_label, color_id = color_test_dataset[idx]
    _, gray_label, gray_id = gray_test_dataset[idx]
    if color_label != gray_label or color_id != gray_id:
        print(f'test dataset is not same: color id: {color_id}, gray id: {gray_id}, color label: {color_label}, gray label: {gray_label}')