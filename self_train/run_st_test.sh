#!/bin/bash

source /home/wzhangbu/anaconda3/etc/profile.d/conda.sh
conda activate st
cd ~/self_train

CUDA_VISIBLE_DEVICES=0 python test_snh_iterative.py
#CUDA_VISIBLE_DEVICES=0,1 python train_snh.py
