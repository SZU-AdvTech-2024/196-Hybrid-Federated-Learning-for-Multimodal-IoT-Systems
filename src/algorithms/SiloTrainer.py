import gc
import random

import torch
import torch.nn as nn
from src.models.ClassfierModel import *
from src.models.MultiModelForCifar import *
from src.algorithms.ClientTrainer import *


class SiloTrainer(object):
    def __init__(self, silo_id, args, train_dataset, test_dataset, batch_size):
        self.silo_id = silo_id
        self.args = args
        self.batch_size = batch_size
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

        self.device = torch.device(self.args.cuda if torch.cuda.is_available() else "cpu")
        self.silo_model = MultiModelForCifar(self.device)
        self.silo_model.to(self.device)
        self.silo_round_num = 2

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.client_num = 2
        self.color_client_num = 1
        self.gray_client_num = 1
        self.client_trainers = {}
        self.client_ids = []
        for i, (dataset_type, color) in enumerate([('Cifar', 1), ('Cifar-gray', 2)]):
            self.client_trainers[i] = ClientTrainer(i, self.args, self.train_dataset, self.test_dataset, self.batch_size, color=color)
            self.client_ids.append(i)

        self.test_loss_list = []
        self.test_acc_rate_list = []

    def print_info(self):
        print(f'    Silo {self.silo_id} Model: ClassifierModel')
        print(f'    Silo {self.silo_id} Round Num: {self.silo_round_num}')
        print(f'    Silo {self.silo_id} Dataset: CIFAR100')
        print(f'    Silo {self.silo_id} Client Num: {self.client_num}')
        for _, client_trainer in self.client_trainers.items():
            client_trainer.print_info()

    def model_aggregate(self, classifier_weight_accumulator, color_weight_accumulator, gray_weight_accumulator):
        print(f'    Silo {self.silo_id} Model Aggregate...')
        for name, param in self.silo_model.classifier.state_dict().items():
            update_per_layer = classifier_weight_accumulator[name] / self.client_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

        for name, param in self.silo_model.color_model.state_dict().items():
            update_per_layer = color_weight_accumulator[name] / self.color_client_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

        for name, param in self.silo_model.gray_model.state_dict().items():
            update_per_layer = gray_weight_accumulator[name] / self.gray_client_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

    def model_train(self):
        self.silo_model.train()
        for batch in self.test_dataloader:
            color, gray, labels, _ = batch
            color, gray, labels = color.to(self.device), gray.to(self.device), labels.to(self.device)
            output = self.silo_model(color, gray)

    def model_eval(self):
        print(f'    Silo {self.silo_id} Model Evaluation...')
        self.model_train()
        self.silo_model.eval()
        total = 0
        correct = 0
        epoch_loss_list = []
        with torch.no_grad():
            for batch in self.test_dataloader:
                color, gray, labels, _ = batch
                color, gray, labels = color.to(self.device), gray.to(self.device), labels.to(self.device)
                output = self.silo_model(color, gray)

                loss = self.criterion(output, labels)
                epoch_loss_list.append(loss.item())
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'    Silo {self.silo_id} test loss avg: {sum(epoch_loss_list) / len(epoch_loss_list)}')
        # print(f'    Silo {self.silo_id} history accuracy on test set: {self.test_acc_rate_list}')
        print(f'    Silo {self.silo_id} accuracy on test set: {(100 * correct / total):.2f}%')
        self.test_acc_rate_list.append(100 * correct / total)
        self.test_loss_list.append(sum(epoch_loss_list) / len(epoch_loss_list))

    def silo_train(self, mini_train_idx):
        classifier_weight_accumulator = {}
        for name, param in self.silo_model.classifier.state_dict().items():
            classifier_weight_accumulator[name] = torch.zeros_like(param)
        color_weight_accumulator = {}
        for name, param in self.silo_model.color_model.state_dict().items():
            color_weight_accumulator[name] = torch.zeros_like(param)
        gray_weight_accumulator = {}
        for name, param in self.silo_model.gray_model.state_dict().items():
            gray_weight_accumulator[name] = torch.zeros_like(param)

        for client_id, client in self.client_trainers.items():
            classifier_diff, color_diff, gray_diff = client.train(self.silo_model, mini_train_idx)
            for name, param in classifier_diff.items():
                classifier_weight_accumulator[name].add_(classifier_diff[name])
            for name, param in color_diff.items():
                color_weight_accumulator[name].add_(color_diff[name])
            for name, param in gray_diff.items():
                gray_weight_accumulator[name].add_(gray_diff[name])

        self.model_aggregate(classifier_weight_accumulator, color_weight_accumulator, gray_weight_accumulator)

    def train(self, global_model, mini_train_idx):
        print(f'    Silo {self.silo_id} model fusion...')
        for name, param in global_model.state_dict().items():
            self.silo_model.state_dict()[name].copy_(param.clone())

        print(f'    Silo {self.silo_id} training...')
        for _ in range(self.silo_round_num):
            self.silo_train(mini_train_idx)

        print(f'    Silo {self.silo_id} test...')
        self.model_eval()

        print(f'    Silo {self.silo_id} create diff dict...')
        classifier_diff = dict()
        for name, param in self.silo_model.classifier.state_dict().items():
            classifier_diff[name] = param - global_model.classifier.state_dict()[name]
        color_diff = dict()
        for name, param in self.silo_model.color_model.state_dict().items():
            color_diff[name] = param - global_model.color_model.state_dict()[name]
        gray_diff = dict()
        for name, param in self.silo_model.gray_model.state_dict().items():
            gray_diff[name] = param - global_model.gray_model.state_dict()[name]
        return classifier_diff, color_diff, gray_diff
