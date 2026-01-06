# from absl import app, flags, logging
# from absl.flags import FLAGS
import torch
import numpy as np
import os
import json
import argparse
import fcntl
from datetime import datetime
from video_engine import process_one_iteration
import pytz
import uuid
import random
from ultralytics import YOLO as YOLO_ultralytics


np.set_printoptions(precision=3)

def read_processed_files(archive_file):
    if not os.path.exists(archive_file):
        return {}
    try:
        with open(archive_file, 'r') as f:
            data = json.load(f) if os.path.getsize(archive_file) > 0 else {}
        return data
    except Exception as e:
        print(f"Error reading archive file: {e}")
        return {}

def write_processed_file(archive_file, file_name, status="completed", error=None):
    try:
        data = read_processed_files(archive_file)
        # Use HK timezone (Asia/Hong_Kong)
        hk_tz = pytz.timezone('Asia/Hong_Kong')
        timestamp = datetime.now(hk_tz).isoformat()
        data[file_name] = {
            "status": status,
            "timestamp": timestamp,
            "error": str(error) if error else None,
            "run_name": args.run_name
        }
        with open(archive_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error writing to archive file: {e}")


def process_video(args, input_file, out_filename, archive_file, file, pose_detector):
    try:
        write_processed_file(archive_file, file, status="processing")
        process_one_iteration(args, input_file, out_filename, pose_detector)
        write_processed_file(archive_file, file, status="completed")
        print(f"=== Finished === \n\n")
    except Exception as e:
        write_processed_file(archive_file, file, status="failed", error=e)
        print(f"=== Failed processing {file}: {str(e)} === \n\n")

def seed_torch(device, seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Video Pose Detection and Pseudo Label Generation")
    
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input video directory')
    parser.add_argument('--out_path', type=str, required=True,
                       help='Path to output directory for processed data')
    parser.add_argument('--model_path', type=str, 
                       default='/home/wzhangbu/elderlycare/weights/yolo11l-pose.pt',
                       help='Path to YOLO pose detection model')
    parser.add_argument('--resume', action='store_true',
                       help='Resume processing from where it left off')
    parser.add_argument('--clip_len', type=int, default=16,
                       help='Number of frames in each clip')
    parser.add_argument('--frame_interval', type=int, default=1,
                       help='Interval between frames to process')
    parser.add_argument('--confidence_threshold', type=float, default=0.90,
                       help='Confidence threshold for pose detection')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Unique name for this run')
    
    return parser.parse_args()

def main(args):
    # Set unique run name if not provided
    if args.run_name is None:
        args.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    print(f"Starting processing with run name: {args.run_name}")
    
    # Initialize pose detector
    pose_detector = YOLO_ultralytics(args.model_path, verbose=False)

    # Create output directory
    os.makedirs(args.out_path, exist_ok=True)

    for day in sorted(os.listdir(args.input_path)):
        input_day = os.path.join(args.input_path, day)
        if not os.path.isdir(input_day):
            continue
            
        print(f"Processing day: {day}")
        
        for camera in sorted(os.listdir(input_day)):
            input_camera = os.path.join(input_day, camera)
            if not os.path.isdir(input_camera):
                continue
                
            print(f"Processing camera: {camera}")
            
            # Setup process tracking
            folder_name = os.path.join(*input_camera.split("/")[-2:])
            archive_file = f"{args.out_path}/{folder_name}/list.json"
            os.makedirs(os.path.dirname(archive_file), exist_ok=True)
            if not args.resume or not os.path.exists(archive_file):
                with open(archive_file, "w") as f:
                    json.dump({}, f)

            # Process videos
            for file in sorted(os.listdir(input_camera)):
                if not file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    continue
                    
                if args.resume:
                    processed_files = read_processed_files(archive_file)
                    if file in processed_files:
                        status = processed_files[file]["status"]
                        if status == "completed":
                            print(f"Skipping already completed file: {file}")
                            continue
                        elif status == "processing":
                            # Skip if being processed by a different run
                            if processed_files[file].get("run_name") != args.run_name:
                                print(f"Skipping file being processed by another run ({processed_files[file].get('run_name')}): {file}")
                                continue

                input_file = os.path.join(input_camera, file)
                out_filename = input_file.split("/")[-3:]  # [day, camera, video_file]
                out_filename = os.path.join(*out_filename).split(".")[0]  # Remove extension
                
                print(f"=== Processing {out_filename} ===")
                os.makedirs(f"{args.out_path}/{os.path.dirname(out_filename)}", exist_ok=True)
                process_video(args, input_file, out_filename, archive_file, file, pose_detector)


if __name__ == '__main__':
    args = parse_args()
    # get a unique id for the run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, args.seed)
    torch.autograd.set_detect_anomaly(True)

    # wandb.require("core")
    # wandb.init(project=args.project, name=args.exp_code)

    main(args)
    print("finished!")

    # wandb.finish()
