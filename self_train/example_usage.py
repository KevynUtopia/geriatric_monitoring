#!/usr/bin/env python3
"""
Example usage of the video pose extraction system.

This script demonstrates how to use the pose extraction system
to process videos and generate YOLO-format pose labels.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pseudo_label_generation import main, parse_args


def create_sample_directory_structure():
    """Create a sample directory structure for testing."""
    sample_dir = Path("sample_videos")
    sample_dir.mkdir(exist_ok=True)
    
    # Create day directories
    day1 = sample_dir / "2023_06_01"
    day2 = sample_dir / "2023_06_02"
    
    # Create camera directories
    (day1 / "camera_1").mkdir(parents=True, exist_ok=True)
    (day1 / "camera_2").mkdir(parents=True, exist_ok=True)
    (day2 / "camera_1").mkdir(parents=True, exist_ok=True)
    
    print(f"Sample directory structure created at: {sample_dir.absolute()}")
    print("Please place your video files (.mp4, .avi, .mov, .mkv) in the camera directories.")
    print("Expected structure:")
    print("sample_videos/")
    print("├── 2023_06_01/")
    print("│   ├── camera_1/")
    print("│   │   ├── video1.mp4")
    print("│   │   └── video2.mp4")
    print("│   └── camera_2/")
    print("│       └── video3.mp4")
    print("└── 2023_06_02/")
    print("    └── camera_1/")
    print("        └── video4.mp4")
    
    return sample_dir


def run_example():
    """Run an example processing session."""
    # Create sample directory structure
    input_dir = create_sample_directory_structure()
    output_dir = Path("sample_output")
    
    print(f"\nExample command to run processing:")
    print(f"python3 pseudo_label_generation.py \\")
    print(f"    --input_path {input_dir} \\")
    print(f"    --out_path {output_dir} \\")
    print(f"    --confidence_threshold 0.95 \\")
    print(f"    --frame_interval 5 \\")
    print(f"    --resume")
    
    print(f"\nOr using the bash script:")
    print(f"./run_pose_extraction.sh \\")
    print(f"    -i {input_dir} \\")
    print(f"    -o {output_dir} \\")
    print(f"    -c 0.95 \\")
    print(f"    -f 5 \\")
    print(f"    -r")


def run_with_custom_args():
    """Run processing with custom arguments."""
    class CustomArgs:
        def __init__(self):
            self.input_path = "sample_videos"
            self.out_path = "sample_output" 
            self.model_path = "yolo11l-pose.pt"
            self.resume = True
            self.clip_len = 16
            self.frame_interval = 5
            self.confidence_threshold = 0.95
            self.seed = 42
            self.run_name = None
    
    args = CustomArgs()
    
    # Check if input directory exists
    if not os.path.exists(args.input_path):
        print(f"Input directory {args.input_path} does not exist.")
        print("Creating sample directory structure...")
        create_sample_directory_structure()
        print("Please add video files to the created directories and run again.")
        return
    
    print("Running pose extraction with custom arguments...")
    try:
        main(args)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error during processing: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example usage of pose extraction system")
    parser.add_argument("--create_sample", action="store_true",
                       help="Create sample directory structure only")
    parser.add_argument("--run_example", action="store_true", 
                       help="Run example processing")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_directory_structure()
    elif args.run_example:
        run_with_custom_args()
    else:
        print("Usage examples for the video pose extraction system:")
        print("=" * 60)
        run_example()
        print("\n" + "=" * 60)
        print("For more options, see:")
        print("python3 pseudo_label_generation.py --help")
        print("./run_pose_extraction.sh --help") 