#!/usr/bin/env bash

# ── Video I/O ──
INPUT_VIDEO="/mnt/storage/July/recording_2019_06_22_9_20_am/cam_11/cam_11-02760-02880.mov"
OUTPUT_VIDEO="/home/wzhangbu/Downloads/output.mp4"

# ── Device ──
DEVICE="cuda:0"

# ── Model Paths ──
YOLO_MODEL="/home/wzhangbu/Desktop/AoE_Demo/snh_demo/backend_weights/yolo11n-pose.pt"
ACTION_CHECKPOINT="/home/wzhangbu/demo_system_backend/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth"

# ── Frame Subsampling ──
FRAME_STEP=8

# ── Detection ──
TOP_IDENTITY=1

# ── Anomaly Detection ──
ANOMALY_THRESHOLD=0.145

python main.py \
    --input              "$INPUT_VIDEO" \
    --output             "$OUTPUT_VIDEO" \
    --device             "$DEVICE" \
    --yolo_model         "$YOLO_MODEL" \
    --checkpoint         "$ACTION_CHECKPOINT" \
    --frame_step         "$FRAME_STEP" \
    --top_identity       "$TOP_IDENTITY" \
    --anomaly_threshold  "$ANOMALY_THRESHOLD"
