#!/usr/bin/env python3
"""
Example script to split the SNH pose dataset.
This script uses the DataSplitter class to split the dataset into train/val/test sets.
"""

import sys
from pathlib import Path

# Add the preprocessing directory to the path
sys.path.append(str(Path(__file__).parent.parent / 'preprocessing'))

from data_split import DataSplitter

def main():
    # Dataset paths (modify these as needed)
    images_dir = 'path_to_your_root/datasets/snh-pose/images'
    labels_dir = 'path_to_your_root/datasets/snh-pose/labels'
    
    # Output directory - creates a new folder for safety
    output_dir = 'path_to_your_root/datasets/snh-pose-split'
    
    # Alternative local output directory for testing
    # output_dir = './snh-pose-split'
    
    print("=" * 60)
    print("SNH Pose Dataset Splitter")
    print("=" * 60)
    print(f"Source images: {images_dir}")
    print(f"Source labels: {labels_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create the data splitter
    splitter = DataSplitter(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        train_ratio=0.7,  # 70% for training
        val_ratio=0.1,    # 10% for validation
        test_ratio=0.2    # 20% for testing
    )
    
    # Run the splitting process (generates file lists)
    # Note: This only creates file lists, not actual copies
    splitter.run_split(max_workers=16, seed=42)
    
    print("\n" + "="*60)
    print("Python splitting completed!")
    print("="*60)
    print("File lists have been generated.")
    print("Next step: Run the bash copying script for efficient file copying.")
    print(f"Execute: {output_dir}/copy_files.sh")
    print("="*60)

if __name__ == "__main__":
    main() 