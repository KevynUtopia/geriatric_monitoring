#!/bin/bash

# Script to run only the copying part after Python splitting is complete
# This script assumes the Python splitting has already been run

echo "========================================"
echo "File Copying Script (Copy Only)"
echo "========================================"

# Set the output directory
OUTPUT_DIR="path_to_your_root/datasets/snh-pose-split"
COPY_SCRIPT="$OUTPUT_DIR/copy_files.sh"

# Check if the copy script exists
if [ ! -f "$COPY_SCRIPT" ]; then
    echo "Error: Copy script not found at: $COPY_SCRIPT"
    echo "Please run the Python data splitting first:"
    echo "  python3 preprocessing/data_split.py [options]"
    echo "  or"
    echo "  ./run_data_split.sh"
    exit 1
fi

# Check if file lists exist
LISTS_DIR="$OUTPUT_DIR/file_lists"
if [ ! -d "$LISTS_DIR" ]; then
    echo "Error: File lists directory not found: $LISTS_DIR"
    echo "Please run the Python data splitting first."
    exit 1
fi

echo "Found copy script: $COPY_SCRIPT"
echo "Found file lists: $LISTS_DIR"
echo ""

# Run the copying script
echo "Starting file copying..."
chmod +x "$COPY_SCRIPT"
bash "$COPY_SCRIPT"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ File copying completed successfully!"
    echo "========================================"
    echo "Output directory: $OUTPUT_DIR"
else
    echo "❌ Error: File copying failed!"
    exit 1
fi 