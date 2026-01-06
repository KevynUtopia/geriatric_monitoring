#!/bin/bash

# Set DATASET_ROOT environment variable before running this script
# Example: export DATASET_ROOT="/path/to/your/dataset"

DATASET_ROOT="/your_root_path"
RUNNING_DIR="results_v6"
RESULT="$DATASET_ROOT/${RUNNING_DIR}_analyze_soft/"
OUT="$DATASET_ROOT/${RUNNING_DIR}_alignment_soft/"
MMACTION2="/home/wzhangbu/elderlycare/mmaction2/"
ACTION="https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth"
#CUDA_VISIBLE_DEVICES=0 python analyze_module/alignment.py \
#      --input_path $RESULT \
#      --output_path $RESULT


for file in $RESULT*; do
  # Check if the file is a regular file
  filename=$(basename "$file")
  echo $filename
#  filename="recording_2019_06_24_8_05_am"
  CUDA_VISIBLE_DEVICES=0 python analyze_module/alignment.py \
      --input_path $RESULT$filename \
      --output_path $OUT$filename \
      --task "alignment_soft_v6" \
      --soft
#      break
done