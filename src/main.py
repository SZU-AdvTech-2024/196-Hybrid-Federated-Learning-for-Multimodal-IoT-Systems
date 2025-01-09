import sys

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import argparse
import datetime

from src.datasets.DataloaderGenerator import *
from src.algorithms.SiloTrainer import *
from src.algorithms.HFMTrainer import *
from src.algorithms.HFLTrainer import *
from src.algorithms.VFLTrainer import *
from src.algorithms.LocalTrainer import *
from src.utils.ExcelUtil import *


def get_args():
    parser.add_argument('--exp_type', type=str, default='Local', help='experiment type: Local, VFL, HFL or HFM')
    parser.add_argument('--cuda', type=str, default='cuda', help='cuda device')
    parser.add_argument('--file_name', type=str, default='result', help='result file name')
    parser.add_argument('--dataset_path', type=str, default='', help='dataset root path')
    parser.add_argument('--round_num', type=int, default=50, help='global round number')
    parser.add_argument('--part_num', type=int, default=10, help='dataset part round number')


parser = argparse.ArgumentParser(description='MMFL')
get_args()
args = parser.parse_args()

if __name__ == '__main__':
    print(f'args: {args}')
    print(f'This is {args.exp_type} Demo')
    global_round_num = args.round_num

    print(f'{args.exp_type} train start...')
    print(f'Global Round Num: {global_round_num}')
    if args.exp_type == 'Local':
        trainer = LocalTrainer(args)
    elif args.exp_type == 'VFL':
        trainer = VFLTrainer(args)
    elif args.exp_type == 'HFL':
        trainer = HFLTrainer(args)
    elif args.exp_type == 'HFM':
        trainer = HFMTrainer(args)
    else:
        trainer = LocalTrainer(args)
    for epoch in range(global_round_num):
        trainer.train(epoch)
    print(f'{args.exp_type} train end...')

    print('Save result...')
    file_head_name = '_' + args.file_name + '_'
    current_time = datetime.datetime.now()
    date_str = current_time.strftime('%Y_%m_%d')

    head_test_loss_lists = trainer.test_loss_list
    excel_file_name = (date_str + file_head_name + 'loss')
    save_acc_to_excel(excel_file_name, head_test_loss_lists, {})

    head_test_acc_rates = trainer.test_acc_rate_list
    excel_file_name = (date_str + file_head_name + 'acc')
    save_acc_to_excel(excel_file_name, head_test_acc_rates, {})
