#!/bin/bash

# Set DATASET_ROOT environment variable before running this script
# Example: export DATASET_ROOT="/path/to/your/dataset"

# Activate conda environment
source /home/wzhangbu/anaconda3/etc/profile.d/conda.sh
conda activate mmaction

# Set working directory
cd ~/elderlycare

# Set paths
DATASET_ROOT="/your_root_path"
DATA_DIR="$DATASET_ROOT/results_v7_alignment_soft/DATASET"
OUTPUT_BASE="$DATASET_ROOT/results_v7_alignment_soft/model_outputs"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_BASE


# Run training script
#CUDA_VISIBLE_DEVICES=3 python time_series_anomaly_detection/train.py
CUDA_VISIBLE_DEVICES=4 python time_series_anomaly_detection/train_lstmad_alpha.py

echo "Training completed. Check the output directory for results."
