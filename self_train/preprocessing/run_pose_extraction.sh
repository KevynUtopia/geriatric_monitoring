#!/bin/bash

source /home/wzhangbu/anaconda3/etc/profile.d/conda.sh
conda activate st
cd ~/self_train

# Video Pose Extraction Script
# This script processes videos to extract frames and generate pose labels

set -e  # Exit on any error

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --input_path PATH        Path to input video directory (required)"
    echo "  -o, --output_path PATH       Path to output directory (required)"
    echo "  -m, --model_path PATH        Path to YOLO pose model (optional)"
    echo "  -r, --resume                 Resume processing from where it left off"
    echo "  -c, --confidence FLOAT       Confidence threshold for pose detection (default: 0.95)"
    echo "  -f, --frame_interval INT     Frame interval to process (default: 1)"
    echo "  -s, --seed INT               Random seed (default: 42)"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -i /path/to/videos -o /path/to/output -r"
    echo ""
    exit 1
}

# Default values
INPUT_PATH="path_to_your_root/July"
OUTPUT_PATH="path_to_your_root/datasets/snh-pose"
MODEL_PATH="/home/wzhangbu/elderlycare/weights/yolo11l-pose.pt"  # Will download if not found
RESUME="--resume"
CONFIDENCE=0.90
FRAME_INTERVAL=300
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input_path)
            INPUT_PATH="$2"
            shift 2
            ;;
        -o|--output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -m|--model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME="--resume"
            shift
            ;;
        -c|--confidence)
            CONFIDENCE="$2"
            shift 2
            ;;
        -f|--frame_interval)
            FRAME_INTERVAL="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
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
if [[ -z "$INPUT_PATH" ]]; then
    echo "Error: Input path is required"
    usage
fi

if [[ -z "$OUTPUT_PATH" ]]; then
    echo "Error: Output path is required"
    usage
fi

# Check if input path exists
if [[ ! -d "$INPUT_PATH" ]]; then
    echo "Error: Input path does not exist: $INPUT_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Print configuration
echo "=============================================="
echo "Video Pose Extraction Configuration"
echo "=============================================="
echo "Input Path:       $INPUT_PATH"
echo "Output Path:      $OUTPUT_PATH"
echo "Model Path:       $MODEL_PATH"
echo "Confidence:       $CONFIDENCE"
echo "Frame Interval:   $FRAME_INTERVAL"
echo "Seed:             $SEED"
echo "Resume:           $([ -n "$RESUME" ] && echo "Yes" || echo "No")"
echo "=============================================="

# Check Python environment
echo "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking required Python packages..."
python3 -c "
import sys
required_packages = ['torch', 'ultralytics', 'opencv-python', 'numpy']
missing_packages = []

for package in required_packages:
    try:
        if package == 'opencv-python':
            import cv2
        else:
            __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'Error: Missing required packages: {missing_packages}')
    print('Please install them using: pip install torch ultralytics opencv-python numpy')
    sys.exit(1)
else:
    print('All required packages are installed.')
"

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU available: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('No GPU available, using CPU')
"

# Run the main script
echo ""
echo "Starting video processing..."
echo "=============================================="

CUDA_VISIBLE_DEVICES=4 python pseudo_label_generation.py \
    --input_path "$INPUT_PATH" \
    --out_path "$OUTPUT_PATH" \
    --model_path "$MODEL_PATH" \
    --confidence_threshold "$CONFIDENCE" \
    --frame_interval "$FRAME_INTERVAL" \
    --seed "$SEED" \
    $RESUME

echo "=============================================="
echo "Processing completed successfully!"
echo "Results saved to: $OUTPUT_PATH"
echo "==============================================" 