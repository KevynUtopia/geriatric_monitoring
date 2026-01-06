# from absl import app, flags, logging
# from absl.flags import FLAGS
from get_args import parse_args

import torch
import numpy as np
import cv2
import wandb
import os
import csv
import pandas as pd

from matplotlib import pyplot as plt
from ultralytics import YOLO as YOLO_ultralytics
# from model.blink_estimate import BlinkEstimatorImageLandmark

from video_engine import train_one_iteration
from utils import seed_torch

np.set_printoptions(precision=3)

def read_file(archive_file):
    with open(archive_file, "r") as f:
        lines = f.readlines()
        previous_files = [line.strip() for line in lines]
    return previous_files

def main(args):
    pose_detector = YOLO_ultralytics(os.path.join(args.weights_path, args.yolo_pose_path), verbose=False)

    # Create CSV file to store results
    csv_path = 'path_to_your_root/results/anno_check.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Check if CSV exists and load processed files
    processed_files = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Create a unique key for each file
                file_key = f"{row['root']}/{row['day']}/{row['cam']}/{row['vid']}"
                processed_files.add(file_key)
        print(f"Found {len(processed_files)} already processed files in CSV")
    else:
        # Create new CSV with headers if it doesn't exist
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['root', 'day', 'cam', 'vid', 'num_humans_detected', 'detection_coordinates'])
        print("Created new CSV file")

    ALL_PROPOSAL_COUNT = 0
    all_days = os.listdir(args.input_path)
    all_days.sort()
    # traverse all days
    for day in all_days:
        input_day = os.path.join(args.input_path, day)
        all_cameras = os.listdir(input_day)
        all_cameras.sort()

        # everyday, traverse all cameras
        for camera in all_cameras:
            input_camera = os.path.join(input_day, camera)

            all_files = os.listdir(input_camera)
            # sort all_files
            all_files.sort()

            # every camera, traverse all videos
            for file in all_files:
                # Check if file has been processed
                file_key = f"{args.input_path}/{day}/{camera}/{file}"
                if file_key in processed_files:
                    continue

                input_file = os.path.join(input_camera, file)
                print(f"=== Processing {input_file} ===")

                # Process video and detect humans
                cap = cv2.VideoCapture(input_file)
                frame_count = 0
                humans_detected = False
                
                while cap.isOpened() and frame_count < 5 and not humans_detected:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue
                    frame_count += 1
                    result = pose_detector(frame, conf=0.50)[0]
                    human_boxes = result.boxes
                    
                    if len(human_boxes) > 0:
                        humans_detected = True
                        num_humans = len(human_boxes)
                        # Get detection coordinates
                        detection_coords = human_boxes.xyxy.cpu().numpy().tolist()
                        # Write results to CSV
                        with open(csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                args.input_path,
                                day,
                                camera,
                                file,
                                num_humans,
                                detection_coords
                            ])
                        print(f"Detected {num_humans} humans in {file}")
                    
                    # Skip 20 frames
                    for _ in range(20):
                        cap.grab()
                
                # If no humans detected after processing all frames, write 0 to CSV
                if not humans_detected:
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            args.input_path,
                            day,
                            camera,
                            file,
                            0,
                            []
                        ])
                    print(f"No humans detected in {file}")
                
                cap.release()
                print(f"=== Finished === \n\n")

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, args.seed)
    torch.autograd.set_detect_anomaly(True)

    main(args)
    print("finished!")
