#!/bin/bash

# Set DATASET_ROOT environment variable before running this script
# Example: export DATASET_ROOT="/path/to/your/dataset"

# cd ~/elderlycare
# bash scripts/run_main0_videomae.sh
source /home/wzhangbu/anaconda3/etc/profile.d/conda.sh
#conda activate smartnursing
conda activate mmaction
cd ~/elderlycare
export YOLO_VERBOSE=True
export OPENCV_LOG_LEVEL=DISABLED

DATASET_ROOT="/your_root_path"
VID="$DATASET_ROOT/July"
OUT="$DATASET_ROOT/results"
GPU_ID=3
RUN_ID=303
SERVER_ID=244
RUN_NAME="${SERVER_ID}_gpu_${GPU_ID}_${RUN_ID}"


MMACTION2="/home/wzhangbu/elderlycare/mmaction2/"
ACTION="https://download.openmmlab.com/mmaction/v1.0/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb_20230314-bf93c9ea.pth"


CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
      --input_path $VID \
      --out_path $OUT \
      --debug \
      --run_name $RUN_NAME \
      --yolo_face_path "yolov11l-face.pt" \
      --yolo_pose_path "yolo11l-pose.pt" \
      --yolo_path "yolo11l.pt" \
      --profile_folder "/home/wzhangbu/elderlycare/people_profile/folder" \
      --frame_interval 20 \
      --clip_len 6 \
      --config $MMACTION2"configs/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py" \
      --checkpoint $ACTION \
      --label_map $MMACTION2"tools/data/ava/re_label_map.txt" \
      --action_score_thr 0.6 \
      --resume \
      --predict_stepsize 60
#      --visualize_video \
#      --re_encode_video '6000k'
#      --resume \
#      --visualize_video \
#      --re_encode_video '10000k'

# /output_video_reid_cam_11-05520-05640.mov
# recording_2019_06_22_9_20_am/cam_11/cam_11-05520-05640.mov