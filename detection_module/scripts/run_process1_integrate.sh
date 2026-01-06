#!/bin/bash

# Set DATASET_ROOT environment variable before running this script
# Example: export DATASET_ROOT="/path/to/your/dataset"

DATASET_ROOT="/your_root_path"
RESULT="$DATASET_ROOT/results_v6"
OUT="$DATASET_ROOT/results_v6_analyze_soft"

CUDA_VISIBLE_DEVICES=0 python analyze_module/analyze.py \
      --input_path $RESULT \
      --output_path $OUT

