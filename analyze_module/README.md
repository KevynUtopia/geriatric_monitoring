# Analysis Module

## Overview

The Analysis Module processes biomarker sequences extracted by the Detection Module to perform advanced analytics, including temporal stability analysis, anomaly detection, and health assessment. It serves as the second stage of the geriatric monitoring pipeline, transforming raw biomarker data into actionable clinical insights.

**Input**: CSV files with biomarker sequences (from detection module)  
**Output**: Analysis results, anomaly alerts, visualizations, and reports

## Key Features

- **Temporal Stability Analysis**: Quantifies biomarker stability patterns over time using factor analysis
- **Anomaly Detection**: Multiple model architectures including LSTM-based and TimeMixer-based anomaly detection
- **Data Preprocessing**: Gaussian smoothing and temporal alignment
- **Visualization**: Gantt charts and temporal heatmaps
- **Evaluation Framework**: Comprehensive evaluation metrics for model performance

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for deep learning models)

### Dependencies

Install required packages:

```bash
pip install pandas numpy scipy scikit-learn matplotlib torch
```

Key dependencies include:
- **PyTorch** - For deep learning models (LSTM, TimeMixer, TimesNet, etc.)
- **Pandas** - For data manipulation
- **NumPy, SciPy** - For numerical operations
- **Scikit-learn** - For factor analysis
- **Matplotlib** - For visualization

## Usage

### Basic Usage

Analyze biomarker sequences:

```bash
python main_system.py \
    --csv_path /path/to/biomarker_sequences.csv \
    --save_path /path/to/results \
    --gantt_chart \
    --sigma 3.0
```

### Key Arguments

- `--csv_path`: Path to input CSV file with biomarker sequences
- `--save_path`: Path to save analysis results
- `--sigma`: Standard deviation for Gaussian filter smoothing (default: 3.0)
- `--gantt_chart`: Generate Gantt chart visualization
- `--color_palette`: Color palette for visualizations

See `integrate_system/arg_parser.py` for complete list of arguments.

## Analysis Pipeline

The module performs the following analysis steps:

1. **Data Loading**: Load biomarker sequences from CSV files
2. **Preprocessing**: Apply Gaussian smoothing and temporal alignment
3. **Factor Analysis**: Extract latent factors and quantify biomarker stability
4. **Anomaly Detection**: Identify deviations from normal patterns using deep learning models
5. **Visualization**: Generate Gantt charts and temporal heatmaps
6. **Alert Generation**: Output timestamps of detected anomalies

## Output Format

The module generates:
- **Alert Timestamps**: CSV file with timestamps of detected anomalies
- **Gantt Charts**: Visual representation of biomarker patterns over time
- **Composite Scores**: Stability scores and anomaly indicators
- **Evaluation Reports**: Performance metrics and analysis results

## Module Files

The `analyze_module/` directory contains several Python scripts for different purposes:

### Core Analysis Scripts

- **`main_system.py`** - Main entry point for the analysis pipeline. Performs factor analysis, anomaly detection, and generates Gantt charts and alert timestamps.

- **`analyze_anomaly_detection.py`** - Batch processing script for anomaly detection analysis. Processes multiple CSV files, applies anomaly detection models, and generates visualization plots.

- **`ts_process.py`** - Time series preprocessing script. Handles timestamp conversion, gap detection, and interpolation for time series data.

### Training and Data Preparation

- **`example_fa_training.py`** - Example script for training Factor Analysis models. Demonstrates how to load train/val/test splits and train person-specific FA models.

- **`create_train_val_test_splits.py`** - Utility script to create train/validation/test data splits. Groups CSV files by person ID, sorts chronologically, and splits data (7:1:2 ratio) for each person.

- **`filter_by_video_intervals.py`** - Script to filter biomarker data based on available video intervals. Extracts valid time periods from video files and filters CSV data accordingly.

### Evaluation Scripts

- **`run_filtered_evaluation.py`** - Main script to run filtered evaluation comparing human annotations with system outputs. Supports identity filtering, date filtering, and comprehensive metrics calculation.

  **What "filtered" means**: The evaluation filters data based on:
  - **Valid time intervals**: Only evaluates time snippets that overlap with actual video recording periods (loaded from `video_time_periods.csv`)
  - **Pre-filtered human annotations**: Uses pre-processed human evaluation data from `human_evaluation_filtered_dir`
  - **Optional filters**: Identity filtering (include/exclude specific persons) and date filtering (specific recording dates)
  - **CSV existence check**: Only evaluates entries where corresponding system output CSV files exist

### Test Scripts

- **`test_csv_existence_checking.py`** - Test script to verify CSV existence checking functionality in the evaluation system. Ensures evaluations only include files that exist.

- **`test_no_csv_evaluation.py`** - Test script to verify evaluation behavior when no CSV files exist. Tests that the system produces zero/null metrics appropriately.

## Evaluation Frameworks

The module includes comprehensive evaluation frameworks:

- **Filtered Evaluation**: `filtered_evaluation/` - Evaluation on filtered datasets. Filters evaluation to only include time snippets within valid video recording intervals, ensuring metrics reflect actual monitoring periods rather than entire time ranges.

- **GCL Evaluation**: `gcl_evaluation/` - Ground-truth comparison evaluation

- **Human Evaluation**: `evaluate/` - Human annotation and system comparison

## Notes

- The module supports various time series model architectures (TimeMixer, TimesNet, LSTM, etc.)
- Factor analysis models can be trained per-person for personalized analysis
- Anomaly detection achieves high performance (AUROC > 0.92) on clinical datasets

## Configuration

### Evaluator Identifiers

The codebase uses the following evaluator identifiers (e1-e5) to reference different human evaluators:

- **e1** = human evaluator 1
- **e2** = human evaluator 2
- **e3** = human evaluator 3
- **e4** = human evaluator 4
- **e5** = human evaluator 5

These identifiers appear in function names (e.g., `read_results_e1()`), file paths, and variable names throughout the module.

### Path Variables

The following path placeholder is used in configuration files and scripts:

- **`path_to_your_analysis_root`**: Root directory path for analysis data and results. Update this to match your local directory structure.

---

*This module is part of the Digital Biomarker Stability Monitoring system. For more information, see the main [README](../README.md).*
