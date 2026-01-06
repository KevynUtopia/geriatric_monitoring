# Detection Module

## Overview

The Detection Module processes video inputs and extracts behavioral biomarker sequences. It converts raw video streams into structured temporal data for health assessment.

**Input**: Video files (`.mov`, `.mp4`, etc.)  
**Output**: CSV files containing timestamped biomarker sequences

## Key Features

- **Human Detection & Tracking**: YOLO-based pose detection and person re-identification (ReID)
- **Action Recognition**: Spatial-temporal action detection using MMAction2 framework
- **Biomarker Extraction**: Tracks 40 visual biomarkers including activities

## Installation

### Prerequisites

- Python 3.7+ (Python 3.8+ recommended)
- CUDA-capable GPU (recommended for video processing)
- CUDA 10.2+ or CUDA 11.8+ (for PyTorch)
- PyTorch 1.8+ (PyTorch 2.4.1 recommended)

### Core Dependencies: MMAction2 and OpenMMLab Ecosystem

**MMAction2** is the most critical dependency for action recognition in this module. It requires **MMEngine** and **MMCV** as essential dependencies, along with optional packages (MMDetection, MMPose) for extended functionality.

**Important**: Please follow the [official MMAction2 installation guide](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) for proper installation. The installation process involves:

1. Installing PyTorch (1.8+)
2. Installing OpenMMLab dependencies (MMEngine, MMCV) using MIM
3. Installing MMAction2 (from source or as a package)
4. Installing additional dependencies from `requirements.txt`

Key dependencies include:
- **MMAction2** - Action recognition framework (most critical)
- **MMEngine & MMCV** - Essential OpenMMLab dependencies
- Ultralytics (YOLO) - Human detection and pose estimation
- OpenCV - Video processing
- MoviePy - Video manipulation
- torchreid - Person re-identification

For version compatibility, ensure `mmengine >= 0.7.2`, `mmcv >= 2.0.0`, and `mmaction2 >= 1.0.0`.

## Usage

The detection module provides several bash scripts in the `scripts/` directory for different stages of the pipeline. Before running any script, set the `DATASET_ROOT` variable in the script to your dataset root path.

### Available Scripts

#### 1. `run_main_videomae.sh` - Video Processing and Biomarker Extraction

**Purpose**: Process video files to extract behavioral biomarker sequences using VideoMAE-based action recognition.

**What it does**:
- Reads video files from input directory
- Detects and tracks humans using YOLO pose detection
- Performs person re-identification (ReID) using BotSort tracker
- Recognizes actions using VideoMAE spatial-temporal models
- Extracts and timestamps biomarker sequences
- Outputs detection results (reid, action, skeleton) as pickle files

#### 2. `run_process1_integrate.sh` - Analysis Phase

**Purpose**: Process raw detection results and generate per-camera person state sequences.

**What it does**:
- Reads detection outputs (reid, action, skeleton pickle files)
- Integrates data from multiple detection sources
- Generates person state sequences per camera
- Outputs CSV files with timestamped biomarker sequences for each person

**Input**: Detection results from `run_main_videomae.sh`  
**Output**: Per-camera person state CSV files

#### 3. `run_process2_alignment.sh` - Alignment Phase

**Purpose**: Align person data across multiple camera views to create unified person profiles.

**What it does**:
- Processes per-camera person state results
- Aligns person identities across different camera views
- Applies soft alignment (mean-based) to merge multi-view data
- Generates final aligned person biomarker sequences

**Input**: Per-camera analysis results from `run_process1_integrate.sh`  
**Output**: Aligned person biomarker sequences

#### 4. `run_main_lstm.sh` - LSTM Anomaly Detection Training

**Purpose**: Train LSTM-based anomaly detection models on biomarker sequences.

**What it does**:
- Loads aligned biomarker sequence data
- Trains LSTM anomaly detection models
- Saves trained model checkpoints

**Input**: Aligned biomarker sequences from `run_process2_alignment.sh`  
**Output**: Trained anomaly detection models

### Pipeline Workflow

The typical workflow follows this sequence:

```
1. run_main_videomae.sh          → Video processing & detection
   ↓
2. run_process1_integrate.sh    → Analysis & integration
   ↓
3. run_process2_alignment.sh    → Multi-camera alignment
   ↓
4. run_main_lstm.sh             → Anomaly detection training
```

## People Profile Directory Structure

The `people_profile/` directory stores person profile images used for person re-identification (ReID). This directory is required for the system to identify and track individuals across video frames.

### Directory Structure

```
people_profile/
├── folder/                      # Main profile directory
│   ├── p_1/                    # Person 1 profile images
│   │   ├── image1.jpg          # Profile image 1
│   │   ├── image2.jpg          # Profile image 2
│   │   └── ...                 # Additional profile images
│   ├── p_2/                    # Person 2 profile images
│   │   └── ...
│   ├── giver_1/                # Caregiver 1 profile images
│   │   └── ...
│   └── ...                     # Additional person profiles
└── test_query/                 # Test query images (optional)
    └── ...
```

### Requirements

- Each subdirectory in `folder/` represents a unique person identifier (e.g., `p_1`, `p_2`, `giver_1`)
- Each person directory should contain multiple profile images (`.jpg`, `.png`, or `.jpeg` format)
- Recommended: 4-10 images per person from different angles and lighting conditions
- Images are used to generate ReID feature representations for person tracking

**Privacy Note**: The `people_profile/folder/` directory in this repository does not contain actual person images to protect privacy. Users must populate this directory with their own profile images following the structure described above.

---

*This module is part of the Digital Biomarker Stability Monitoring system. For more information, see the main [README](../README.md).*
