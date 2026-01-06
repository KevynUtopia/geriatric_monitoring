#!/bin/bash

# SNH Pose Dataset Splitting Script
# This script splits the dataset into train/val/test sets with a 7:1:2 ratio

echo "========================================"
echo "SNH Pose Dataset Splitting Script"
echo "========================================"

# Set the paths
IMAGES_DIR="path_to_your_root/datasets/snh-pose/images"
LABELS_DIR="path_to_your_root/datasets/snh-pose/labels"
OUTPUT_DIR="path_to_your_root/datasets/snh-pose-split"

# Check if source directories exist
if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found: $IMAGES_DIR"
    exit 1
fi

if [ ! -d "$LABELS_DIR" ]; then
    echo "Error: Labels directory not found: $LABELS_DIR"
    exit 1
fi

echo "Source directories:"
echo "  Images: $IMAGES_DIR"
echo "  Labels: $LABELS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Step 1: Run the Python data splitting (generates file lists)
echo "Step 1: Running Python data splitting to generate file lists..."
python3 preprocessing/data_split.py \
    --images_dir "$IMAGES_DIR" \
    --labels_dir "$LABELS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --test_ratio 0.2 \
    --max_workers 16 \
    --seed 42

# Check if Python script succeeded
if [ $? -ne 0 ]; then
    echo "Error: Python data splitting failed!"
    exit 1
fi

echo ""
echo "Python data splitting completed!"
echo "File lists generated successfully."

# Step 2: Run the bash copying script
echo ""
echo "Step 2: Running bash copying script..."
COPY_SCRIPT="$OUTPUT_DIR/copy_files.sh"

if [ -f "$COPY_SCRIPT" ]; then
    chmod +x "$COPY_SCRIPT"
    bash "$COPY_SCRIPT"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "✅ Data splitting and copying completed successfully!"
        echo "========================================"
        echo "Output directory: $OUTPUT_DIR"
        
        # Show final directory structure
        if [ -d "$OUTPUT_DIR" ]; then
            echo ""
            echo "Final directory structure:"
            tree "$OUTPUT_DIR" -d -L 2 2>/dev/null || ls -la "$OUTPUT_DIR"
        fi
    else
        echo "Error: Bash copying script failed!"
        exit 1
    fi
else
    echo "Error: Copying script not found: $COPY_SCRIPT"
    echo "Please check the Python script output for errors."
    exit 1
fi 