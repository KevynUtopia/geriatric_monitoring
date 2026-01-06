#!/bin/bash

source /home/wzhangbu/anaconda3/etc/profile.d/conda.sh
conda activate st
cd ~/self_train

# YOLO11 Keypoint Detection Training Script

set -e  # Exit on any error

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --data PATH              Path to dataset config file (required)"
    echo "  -m, --model MODEL            Model name (default: yolo11n-pose.pt)"
    echo "  -e, --epochs INT             Number of epochs (default: 100)"
    echo "  -b, --batch_size INT         Batch size (default: 16)"
    echo "  -c, --config PATH            Config file path (default: config/config.yaml)"
    echo "  --mode MODE                  Training mode: train|self_train|evaluate (default: train)"
    echo "  --unlabeled PATH             Unlabeled data path (for self-training)"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -d data/coco/dataset.yaml -e 200 -b 32"
    echo "  $0 -d data/dataset.yaml --mode self_train --unlabeled data/unlabeled"
    echo ""
    exit 1
}

# Default values
DATA_CONFIG="config/config.yaml"
MODEL="yolo11n-pose.pt"
EPOCHS=100
BATCH_SIZE=16
CONFIG_FILE="config/config.yaml"
MODE="train"
UNLABELED_DATA=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data)
            DATA_CONFIG="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --unlabeled)
            UNLABELED_DATA="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATA_CONFIG" ]]; then
    echo "Error: Data config path is required"
    usage
fi

# Check if data config exists
if [[ ! -f "$DATA_CONFIG" ]]; then
    echo "Error: Data config file does not exist: $DATA_CONFIG"
    exit 1
fi

# Validate mode-specific requirements
if [[ "$MODE" == "self_train" && -z "$UNLABELED_DATA" ]]; then
    echo "Error: Unlabeled data path is required for self-training mode"
    exit 1
fi

# Build command based on mode
if [[ "$MODE" == "train" ]]; then
    CUDA_VISIBLE_DEVICES=4 python train.py \
        --mode train \
        --data "$DATA_CONFIG" \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --config "$CONFIG_FILE"

elif [[ "$MODE" == "self_train" ]]; then
    CUDA_VISIBLE_DEVICES=4 python train.py \
        --mode self_train \
        --data "$DATA_CONFIG" \
        --unlabeled_data "$UNLABELED_DATA" \
        --model "$MODEL" \
        --config "$CONFIG_FILE"

elif [[ "$MODE" == "evaluate" ]]; then
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --mode evaluate \
        --data "$DATA_CONFIG" \
        --model "$MODEL" \
        --config "$CONFIG_FILE"

else
    echo "Error: Invalid mode '$MODE'. Use: train, self_train, or evaluate"
    exit 1
fi 