# Data Splitting for SNH Pose Dataset

This repository contains efficient tools for splitting large datasets into train/validation/test sets. The implementation is optimized for handling ~160k images with parallel processing.

## Overview

The data splitting tool uses a hybrid approach:
1. **Python Phase**: Scans image directory to extract all base names
2. **Python Phase**: Randomly splits them into train/val/test sets (7:1:2 ratio by default)
3. **Python Phase**: Creates organized directory structure and file lists
4. **Bash Phase**: Copies files efficiently using rsync/cp with parallel processing
5. **Bash Phase**: Verifies the split integrity

## Directory Structure

After splitting, the output directory will have this structure:
```
output_directory/
├── images/
│   ├── train/          # 70% of images
│   ├── val/            # 10% of images
│   └── test/           # 20% of images
├── labels/
│   ├── train/          # 70% of labels
│   ├── val/            # 10% of labels
│   └── test/           # 20% of labels
├── file_lists/         # Generated file lists for copying
│   ├── train_images.txt
│   ├── train_labels.txt
│   ├── val_images.txt
│   ├── val_labels.txt
│   ├── test_images.txt
│   └── test_labels.txt
└── copy_files.sh       # Auto-generated copying script
```

## Usage

### Method 1: Using the Shell Script (Recommended)

The easiest way to run the data splitting:

```bash
./run_data_split.sh
```

This script:
- Uses the predefined paths for SNH pose dataset
- Applies 7:1:2 train/val/test split
- Runs Python splitting to generate file lists
- Runs bash copying script for efficient file transfer
- Sets random seed to 42 for reproducibility

### Method 2: Using Python Script Directly

For more control over parameters:

```bash
python3 preprocessing/data_split.py \
    --images_dir "path_to_your_root/datasets/snh-pose/images" \
    --labels_dir "path_to_your_root/datasets/snh-pose/labels" \
    --output_dir "path_to_your_root/datasets/snh-pose-split" \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --test_ratio 0.2 \
    --max_workers 16 \
    --seed 42
```

### Method 3: Using the Example Script

```bash
python3 examples/split_snh_pose_data.py
```

### Method 4: Copy Files Only (After Python Splitting)

If you've already run the Python splitting and want to copy files only:

```bash
./run_copy_only.sh
```

### Method 5: Programmatic Usage

```python
from preprocessing.data_split import DataSplitter

# Create splitter instance
splitter = DataSplitter(
    images_dir="/path/to/images",
    labels_dir="/path/to/labels", 
    output_dir="/path/to/output",
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2
)

# Run the splitting (generates file lists only)
splitter.run_split(max_workers=16, seed=42)

# Then manually run the bash copying script:
# bash /path/to/output/copy_files.sh
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `images_dir` | Path to source images directory | Required |
| `labels_dir` | Path to source labels directory | Required |
| `output_dir` | Path to output directory | Required |
| `train_ratio` | Proportion for training set | 0.7 |
| `val_ratio` | Proportion for validation set | 0.1 |
| `test_ratio` | Proportion for test set | 0.2 |
| `max_workers` | Number of parallel workers | 4 |
| `seed` | Random seed for reproducibility | 42 |

## Features

### Efficiency Optimizations
- **Hybrid Approach**: Python for logic, Bash for file operations
- **Optimized File Transfer**: Uses batch processing with xargs for efficient copying with fallback to individual cp
- **Parallel Processing**: Concurrent copying of images and labels
- **Progress Tracking**: Real-time progress bars with `tqdm` and bash progress indicators
- **Memory Efficient**: Streams file operations without loading all data into memory
- **Error Handling**: Graceful handling of file operation errors

### Safety Features
- **Validation**: Ensures train/val/test ratios sum to 1.0
- **Verification**: Counts files in each split to verify integrity
- **Error Reporting**: Detailed error messages for debugging
- **Dry Run Capability**: Creates new directories without affecting originals

### File Processing
- **Flexible Extensions**: Supports .jpg, .jpeg, .png, .bmp, .tiff (case-insensitive)
- **Direct Base Name Processing**: Extracts base names from images and assumes corresponding labels exist
- **Extension Detection**: Automatically detects the correct image extension for each base name

## Expected Performance

For ~160k images:
- **Python Phase**: ~1-3 minutes (scanning and file list generation)
- **Bash Phase**: ~8-25 minutes (depends on file sizes and storage speed)
- **Verification**: ~1-2 minutes

**Benefits of Hybrid Approach:**
- Eliminates Python threading bottlenecks
- Uses system-optimized batch file transfer with xargs
- Copies files directly to target folders (preserves only filenames)
- Better handling of large file operations
- More robust for network storage (NFS)

## File Naming Convention

The tool expects:
- **Images**: `*.jpg`, `*.jpeg`, `*.png`, `*.bmp`, `*.tiff`
- **Labels**: `*.txt` with same base name as image

Example:
```
recording_2019_06_22_9_20_am_cam_11_cam_11-00000-00120_000000000000.jpg
recording_2019_06_22_9_20_am_cam_11_cam_11-00000-00120_000000000000.txt
```

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure read access to source directories and write access to output directory
2. **Storage Space**: Verify sufficient space (splitting doubles storage requirement)
3. **Network Storage**: For NFS/network storage, consider reducing `max_workers` if experiencing timeouts
4. **Missing Files**: Check the console output for base names without corresponding image or label files

### Performance Tuning

- **More Workers**: Increase `max_workers` for faster copying (8-32 depending on system)
- **Local Storage**: Copy to local storage first for faster processing
- **SSD Storage**: Use SSD storage for better I/O performance

## Example Output

### Python Phase:
```
Starting data split with seed: 42
Source directories:
  Images: path_to_your_root/datasets/snh-pose/images
  Labels: path_to_your_root/datasets/snh-pose/labels
Output directory: path_to_your_root/datasets/snh-pose-split
Split ratios: 0.7:0.1:0.2
--------------------------------------------------
Scanning for image files to get base names...
Found 160000 image files
Extracted 160000 base names
Splitting data into train/val/test sets...
Split sizes:
  Train: 112000 files (70.0%)
  Val: 16000 files (10.0%)
  Test: 32000 files (20.0%)
Generating file lists for bash copying...
Generating train file lists...
  train: 112000 images, 112000 labels
Generating val file lists...
  val: 16000 images, 16000 labels
Generating test file lists...
  test: 32000 images, 32000 labels
File lists written to: /nfs/.../snh-pose-split/file_lists
Bash copying script created: /nfs/.../snh-pose-split/copy_files.sh
--------------------------------------------------
Data splitting completed!
Total time: 127.45 seconds
Total files processed: 160000 base names
Output directory: path_to_your_root/datasets/snh-pose-split

==================================================
NEXT STEP: Run the bash copying script
==================================================
Execute: /nfs/.../snh-pose-split/copy_files.sh
...
```

### Bash Phase:
```
========================================
File Copying Script for Data Splitting
========================================
Starting file copying process...
Using batch processing for efficient copying...

=== Copying TRAIN set ===
Copying 112000 images files to train set using batch processing...
Copying 112000 labels files to train set using batch processing...
...

=== Copying VALIDATION set ===
...

=== Copying TEST set ===
...

========================================
File copying completed!
========================================

Verifying split...
train set: 112000 images, 112000 labels
val set: 16000 images, 16000 labels
test set: 32000 images, 32000 labels

Data splitting and copying completed successfully!
```

## Requirements

- Python 3.7+
- tqdm (for progress bars)
- Standard library modules: os, random, pathlib, time, stat
- Bash shell with xargs and cp

All Python requirements are included in the main `requirements.txt` file. 