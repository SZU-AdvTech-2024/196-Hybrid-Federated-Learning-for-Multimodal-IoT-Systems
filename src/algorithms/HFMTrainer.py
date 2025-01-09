import torch
import torch.nn as nn
from src.models.ClassfierModel import *
from src.models.MultiModelForCifar import *
from src.algorithms.SiloTrainer import *


class HFMTrainer(object):
    def __init__(self, args):
        self.args = args
        self.batch_size = 128
        self.train_dataset, self.test_dataset = generate_dataset('Multiple', self.args.dataset_path)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)
        self.train_idx = []
        self.split_train_dataset_index()

        self.device = torch.device(self.args.cuda if torch.cuda.is_available() else "cpu")
        self.global_model = MultiModelForCifar(self.device)
        self.global_model.to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.silo_num = 5
        self.silo_trainers = {}
        self.silo_ids = []
        for i in range(self.silo_num):
            self.silo_trainers[i] = SiloTrainer(i, self.args, self.train_dataset, self.test_dataset, self.batch_size)
            self.silo_ids.append(i)

        self.test_loss_list = []
        self.test_acc_rate_list = []
        self.print_info()

    def split_train_dataset_index(self):
        all_idx = list(range(len(self.train_dataset)))
        part_num = self.args.part_num
        random.shuffle(all_idx)
        part_size = len(all_idx) // part_num
        remainder = len(all_idx) % part_num
        start = 0

        for i in range(part_num):
            end = start + part_size + (1 if i < remainder else 0)
            self.train_idx.append(all_idx[start:end])
            start = end

    def print_info(self):
        print(f'HFM Device: {self.device}')
        print(f'HFM Silo Num: {self.silo_num}')
        print(f'HFM Dataset: CIFAR100')
        for _, silo_trainer in self.silo_trainers.items():
            silo_trainer.print_info()

    def model_aggregate(self, classifier_weight_accumulator, color_weight_accumulator, gray_weight_accumulator):
        print(f'    HFM Model Aggregate...')
        for name, param in self.global_model.classifier.state_dict().items():
            update_per_layer = classifier_weight_accumulator[name] / self.silo_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

        for name, param in self.global_model.color_model.state_dict().items():
            update_per_layer = color_weight_accumulator[name] / self.silo_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

        for name, param in self.global_model.gray_model.state_dict().items():
            update_per_layer = gray_weight_accumulator[name] / self.silo_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

    def model_train(self):
        self.global_model.train()
        for batch in self.test_dataloader:
            color, gray, labels, _ = batch
            color, gray, labels = color.to(self.device), gray.to(self.device), labels.to(self.device)
            output = self.global_model(color, gray)

    def model_eval(self):
        print(f'    HFM Model Evaluation...')
        self.model_train()
        self.global_model.eval()
        total = 0
        correct = 0
        epoch_loss_list = []
        with torch.no_grad():
            for batch in self.test_dataloader:
                color, gray, labels, _ = batch
                color, gray, labels = color.to(self.device), gray.to(self.device), labels.to(self.device)
                output = self.global_model(color, gray)

                loss = self.criterion(output, labels)
                epoch_loss_list.append(loss.item())
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'    HFM test loss avg: {sum(epoch_loss_list) / len(epoch_loss_list)}')
        print(f'    HFM history accuracy on test set: {self.test_acc_rate_list}')
        print(f'    HFM accuracy on test set: {(100 * correct / total):.2f}%')
        self.test_acc_rate_list.append(100 * correct / total)
        self.test_loss_list.append(sum(epoch_loss_list) / len(epoch_loss_list))

    def train(self, epoch):
        print(f'    HFM train epoch: {epoch}')
        mini_train_idx = self.train_idx[epoch % len(self.train_idx)]

        classifier_weight_accumulator = {}
        for name, param in self.global_model.classifier.state_dict().items():
            classifier_weight_accumulator[name] = torch.zeros_like(param)
        color_weight_accumulator = {}
        for name, param in self.global_model.color_model.state_dict().items():
            color_weight_accumulator[name] = torch.zeros_like(param)
        gray_weight_accumulator = {}
        for name, param in self.global_model.gray_model.state_dict().items():
            gray_weight_accumulator[name] = torch.zeros_like(param)

        for silo_id, silo in self.silo_trainers.items():
            classifier_diff, color_diff, gray_diff = silo.train(self.global_model, mini_train_idx)
            for name, param in classifier_diff.items():
                classifier_weight_accumulator[name].add_(classifier_diff[name])
            for name, param in color_diff.items():
                color_weight_accumulator[name].add_(color_diff[name])
            for name, param in gray_diff.items():
                gray_weight_accumulator[name].add_(gray_diff[name])

        self.model_aggregate(classifier_weight_accumulator, color_weight_accumulator, gray_weight_accumulator)
        self.model_eval()
