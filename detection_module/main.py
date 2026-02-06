# from absl import app, flags, logging
# from absl.flags import FLAGS
from get_args import parse_args
import torch
import numpy as np
import os
import json
import fcntl
from datetime import datetime
from video_engine import train_one_iteration
from utils import seed_torch
import pytz
import uuid

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


def process_video(args, input_file, out_filename, archive_file, file):

    train_one_iteration(args, input_file, out_filename)
    try:
        write_processed_file(archive_file, file, status="processing")
        train_one_iteration(args, input_file, out_filename)
        write_processed_file(archive_file, file, status="completed")
        print(f"=== Finished === \n\n")
    except Exception as e:
        write_processed_file(archive_file, file, status="failed", error=e)
        print(f"=== Failed processing {file}: {str(e)} === \n\n")

def main(args):

    for day in sorted(os.listdir(args.input_path)):
        input_day = os.path.join(args.input_path, day)
        # input_day = 'path_to_your_root/July/recording_2019_06_27_10_30_am'
        # if '07_1' in input_day:
        #     print(f"Skipping {input_day}")
        #     continue
        for camera in sorted(os.listdir(input_day)):
            input_camera = os.path.join(input_day, camera)

            # DEBUG
            # input_camera = 'path_to_your_root/July/recording_2019_06_27_10_30_am/cam_13'
            
            # Setup process tracking
            folder_name = os.path.join(*input_camera.split("/")[-2:])
            archive_file = f"{args.out_path}/{folder_name}/list.json"
            os.makedirs(os.path.dirname(archive_file), exist_ok=True)
            if not args.resume or not os.path.exists(archive_file):
                with open(archive_file, "w") as f:
                    json.dump({}, f)

            # Process videos
            for file in sorted(os.listdir(input_camera)):
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
                out_filename = input_file.split("July/")[1].split(".")[0]
                
                print(f"=== Processing {out_filename} ===")
                os.makedirs(f"{args.out_path}/{os.path.dirname(out_filename)}", exist_ok=True)
                process_video(args, input_file, out_filename, archive_file, file)

                # exit(0)
        # exit(0)


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
