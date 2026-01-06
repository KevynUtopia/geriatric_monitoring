#!/bin/bash

# AI Detection Template - Run Script
# This script sets up the proper environment and runs the application

echo "🚀 Starting AI Detection Template..."

# Activate conda environment
echo "📦 Activating conda environment: snh_demo"
source ~/opt/anaconda3/etc/profile.d/conda.sh
conda activate snh_demo

# Set OpenCV environment variable to skip camera authorization
export OPENCV_AVFOUNDATION_SKIP_AUTH=1

# Run the application
echo "🎯 Launching YOLO Detection Application..."
python run.py

echo "👋 Application closed."
