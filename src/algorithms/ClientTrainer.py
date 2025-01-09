import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from src.models.ClassfierModel import *
from src.models.Resnet18ForModelNet import *
from src.models.ResNetForMNIST import *
from src.models.CNNForCifar import *
from src.models.MultiModelForCifar import *
from src.datasets.DataloaderGenerator import *


class ClientTrainer:
    def __init__(self, client_id, args, train_dataset, test_dataset, batch_size, local_round_num=5, learning_rate=0.001, color=1):
        self.client_id = client_id
        self.args = args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        self.local_round_num = local_round_num
        self.learning_rate = learning_rate
        self.weight_decay = 0.001
        self.prox_lamda = 0.01
        self.color = color

        self.device = torch.device(self.args.cuda if torch.cuda.is_available() else "cpu")
        self.model = MultiModelForCifar(self.device)
        self.model.to(self.device)

        classifier_params = self.model.classifier.parameters()
        color_params = self.model.color_model.parameters()
        gray_params = self.model.gray_model.parameters()

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        if self.color == 1:
            self.optimizer = optim.Adam([
                {'params': classifier_params, 'weight_decay': self.weight_decay},
                {'params': color_params, 'weight_decay': self.weight_decay},
                {'params': gray_params, 'weight_decay': self.weight_decay, 'lr': 0}
            ], lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.color == 2:
            self.optimizer = optim.Adam([
                {'params': classifier_params, 'weight_decay': self.weight_decay},
                {'params': color_params, 'weight_decay': self.weight_decay, 'lr': 0},
                {'params': gray_params, 'weight_decay': self.weight_decay}
            ], lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.color == 3:
            self.optimizer = optim.Adam([
                {'params': classifier_params, 'weight_decay': self.weight_decay},
                {'params': color_params, 'weight_decay': self.weight_decay},
                {'params': gray_params, 'weight_decay': self.weight_decay}
            ], lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150, eta_min=0.00001)

        self.client_test_loss_list = []
        self.client_test_acc_rate_list = []

    def compute_prox_loss(self, global_model):
        prox_loss = 0
        for global_param, local_param in zip(global_model.parameters(), self.model.parameters()):
            prox_loss += ((global_param - local_param)**2).sum()
        return prox_loss

    def print_info(self):
        print(f'        Client ID: {self.client_id}')
        print(f'        Client {self.client_id} Device: {self.device}')
        print(f'        Client {self.client_id} Model: MultiModelForCifar')
        print(f'        Client {self.client_id} Local Round Num: {self.local_round_num}')
        print(f'        Client {self.client_id} Dataset Color: {self.color}')
        print(f'        Client {self.client_id} Dataset Batch Size: {self.batch_size}')
        print(f'        Client {self.client_id} Learning Rate: {self.learning_rate}')

    def train(self, global_model, mini_train_idx):
        # 小批量数据集生成
        train_dataloader = generate_mini_dataloader(self.train_dataset, self.batch_size, mini_train_idx)
        # 模型聚合
        # print(f'        Client {self.client_id} model fusion...')
        for name, param in global_model.state_dict().items():
            self.model.state_dict()[name].copy_(param.clone())

        # print(f'        Client {self.client_id} train...')
        self.model.train()
        for epoch in range(self.local_round_num):
            for batch in train_dataloader:
                color, gray, labels, _ = batch
                color, gray, labels = color.to(self.device), gray.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(color, gray)
                loss = self.criterion(output, labels)
                prox_loss = self.compute_prox_loss(global_model)
                loss += self.prox_lamda * prox_loss
                loss.backward()
                self.optimizer.step()
            # self.scheduler.step()

        # print(f'        Client {self.client_id} test...')
        self.model.eval()
        total = 0
        correct = 0
        epoch_test_loss_list = []
        with torch.no_grad():
            for batch in self.test_dataloader:
                color, gray, labels, _ = batch
                color, gray, labels = color.to(self.device), gray.to(self.device), labels.to(self.device)
                output = self.model(color, gray)
                loss = self.criterion(output, labels)
                prox_loss = self.compute_prox_loss(global_model)
                loss += self.prox_lamda * prox_loss
                epoch_test_loss_list.append(loss.item())

                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # print(f'        Client {self.client_id} test loss avg: {sum(epoch_test_loss_list) / len(epoch_test_loss_list)}')
        # print(f'        Client {self.client_id} history accuracy on test set: {self.client_test_acc_rate_list}')
        # print(f'        Client {self.client_id} accuracy on test set: {(100 * correct / total):.2f}%')
        self.client_test_acc_rate_list.append(100 * correct / total)
        self.client_test_loss_list.append(sum(epoch_test_loss_list) / len(epoch_test_loss_list))

        # print(f'        Client {self.client_id} create diff dict...')
        classifier_diff = dict()
        for name, param in self.model.classifier.state_dict().items():
            classifier_diff[name] = param - global_model.classifier.state_dict()[name]
        color_diff = dict()
        for name, param in self.model.color_model.state_dict().items():
            color_diff[name] = param - global_model.color_model.state_dict()[name]
        gray_diff = dict()
        for name, param in self.model.gray_model.state_dict().items():
            gray_diff[name] = param - global_model.gray_model.state_dict()[name]
        return classifier_diff, color_diff, gray_diff

