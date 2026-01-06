# Combined Analysis and Alignment Pipeline

This module provides a unified script that combines the functionality of `analyze.py`, `analyzor.py`, and `alignment.py` to directly produce aligned results from elderlycare video data.

## Overview

The pipeline consists of two main phases:

1. **Analysis Phase**: Processes raw video data (reid, action, skeleton) and generates per-camera person state results
2. **Alignment Phase**: Aligns person data across multiple camera views to produce comprehensive aligned results

## Files

- `analyze_and_align.py` - Main combined script
- `analyze.py` - Original analysis script (standalone)
- `analyzor.py` - Analyzer class definition
- `alignment.py` - Original alignment script (standalone)
- `README.md` - This documentation

## Usage

### Using the Python Script Directly

```bash
python analyze_module/analyze_and_align.py \
    --input_path /path/to/input/data \
    --output_path /path/to/output/results \
    --alignment_task alignment \
    --soft
```

### Using the Shell Script (Recommended)

```bash
# Basic usage
./scripts/run_analyze_and_align.sh -i /path/to/input -o /path/to/output

# With soft alignment
./scripts/run_analyze_and_align.sh -i /path/to/input -o /path/to/output --soft

# Skip analysis phase (if already completed)
./scripts/run_analyze_and_align.sh --skip-analysis -o /path/to/output

# Custom alignment task name
./scripts/run_analyze_and_align.sh -t alignment_soft_v6 --soft
```

## Arguments

### Required Arguments
- `--input_path`: Path to input data containing recording folders
- `--output_path`: Path where results will be saved

### Optional Arguments
- `--alignment_task`: Name for alignment output folder (default: "alignment")
- `--soft`: Use soft alignment (mean) instead of mode-based alignment
- `--skip_analysis`: Skip the analysis phase if results already exist
- `--skip_alignment`: Skip the alignment phase
- `--help`: Show help message

## Input Data Structure

The input data should be organized as follows:

```
input_path/
├── recording_2019_06_22_9_20_am/
│   ├── cam_10/
│   │   ├── list.json
│   │   ├── video1-action.pkl
│   │   ├── video1-123456-reid.pkl
│   │   ├── video1-skeleton.pkl
│   │   └── ...
│   ├── cam_11/
│   │   ├── list.json
│   │   └── ...
│   └── ...
├── recording_2019_06_24_8_05_am/
│   └── ...
└── ...
```

## Output Structure

The output will be organized as follows:

```
output_path/
├── recording_2019_06_22_9_20_am/
│   ├── cam_10/
│   │   ├── person1.csv
│   │   ├── person1_skeleton.pt
│   │   ├── person2.csv
│   │   ├── person2_skeleton.pt
│   │   └── ...
│   ├── cam_11/
│   │   └── ...
│   └── alignment/  # or custom alignment task name
│       ├── person1.csv
│       ├── person2.csv
│       └── ...
├── recording_2019_06_24_8_05_am/
│   └── ...
└── ...
```

## Pipeline Workflow

### Phase 1: Analysis
1. Iterates through all recording days
2. For each day, processes all cameras
3. For each camera, processes all videos listed in `list.json`
4. Loads reid, action, and skeleton data for each video
5. Updates person states using the Analyzor class
6. Saves individual person results as CSV files and skeleton data

### Phase 2: Alignment
1. Identifies all processed days from Phase 1
2. For each day, finds all cameras and person candidates
3. Aligns person data across multiple camera views
4. Applies either mode-based (default) or mean-based (soft) alignment
5. Saves final aligned results

## Features

- **Error Handling**: Robust error handling with informative warnings
- **Resume Capability**: Can skip already processed data
- **Progress Tracking**: Uses tqdm for progress bars
- **Flexible Alignment**: Supports both hard (mode) and soft (mean) alignment
- **Validation**: Verifies data consistency across reid, action, and skeleton files
- **Modular Design**: Can run phases independently

## Examples

### Complete Pipeline
```bash
# Process everything from scratch
./scripts/run_analyze_and_align.sh \
    -i path_to_your_root/results_v2 \
    -o path_to_your_root/results_combined \
    --soft \
    -t alignment_soft_v6
```

### Analysis Only
```bash
# Only run analysis phase
./scripts/run_analyze_and_align.sh \
    -i /path/to/input \
    -o /path/to/output \
    --skip-alignment
```

### Alignment Only
```bash
# Only run alignment phase (analysis already completed)
./scripts/run_analyze_and_align.sh \
    -o /path/to/output \
    --skip-analysis \
    --soft
```

## Dependencies

- pandas
- tqdm
- pickle
- json
- torch (for skeleton data)
- The original `Analyzor` class and `Alignment_Worker` class

## Notes

- The script automatically handles CUDA device selection (defaults to GPU 0)
- Time consistency is maintained with a tolerance of ±3 time units
- Unknown identities are filtered out during processing
- Requires minimum 2 cameras for alignment to proceed
- Progress is saved incrementally, allowing for resumption if interrupted 