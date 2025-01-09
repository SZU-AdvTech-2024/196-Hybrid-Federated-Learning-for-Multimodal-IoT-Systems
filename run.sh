#!/bin/bash


cd "$(dirname "$0")" || exit
cd src || exit

echo "Run experiment scripts"

echo "Local test"
python main.py --exp_type=Local --cuda=cuda:2 --file_name=local_resnet_cifar100_big_batch_prox --part_num=5
if [ $? -ne 0 ]; then
    echo "Local test failed."
    exit 1
fi

echo "VFL test"
python main.py --exp_type=VFL --cuda=cuda:2 --file_name=vfl_resnet_cifar100_big_batch_prox --part_num=5
if [ $? -ne 0 ]; then
    echo "hfl_test.py failed."
    exit 1
fi


echo "HFL test"
python main.py --exp_type=HFL --cuda=cuda:2 --file_name=hfl_resnet_cifar100_big_batch_prox --part_num=5
if [ $? -ne 0 ]; then
    echo "hfl_test.py failed."
    exit 1
fi

echo "HFM test"
python main.py --exp_type=HFM --cuda=cuda:2 --file_name=hfm_resnet_cifar100_big_batch_prox --part_num=5
if [ $? -ne 0 ]; then
    echo "HFM test failed."
    exit 1
fi

echo "All scripts ran successfully."

# 返回上级目录（即最初的脚本所在目录）
cd..