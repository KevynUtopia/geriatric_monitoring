#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4 PORT=29666 bash tools/dist_train.sh  \
    configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py \
    5 \
    --seed 0