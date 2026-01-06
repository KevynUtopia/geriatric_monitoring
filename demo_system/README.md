# Demo System

This directory contains a **standalone demonstration** of the Digital Biomarker Stability Monitoring system for privacy-preserving geriatric care. This demo showcases the core functionality of the entire repository, including video processing, biomarker extraction, and real-time analysis capabilities.

## Overview

The demo system provides an interactive interface to experience the geriatric monitoring system without requiring the full repository setup. It demonstrates:

- **Real-time video processing** - Live camera feed or video file analysis
- **Human detection and tracking** - YOLO-based pose detection and person re-identification
- **Action recognition** - Spatial-temporal action detection using MMAction2
- **Biomarker extraction** - Visual biomarker sequence generation
- **Interactive visualization** - Real-time display of detection results and statistics

## Standalone Usage

**You can download and run this `demo_system` folder independently** without needing the entire repository. This makes it easy to quickly explore the system's capabilities.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- Webcam or video file for input

## Installation

1. **Clone or download this `demo_system` folder**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights:**
   
   The required model weights are available in the following Google Drive folder:
   
   **[Download Models](https://drive.google.com/drive/folders/1TuGWuag2kme74HmJ-u5t9l6wWkFJ45sc?usp=sharing)**
   

## Running the Demo

### Option 1: Using the run script (recommended)
```bash
bash run_app.sh
```

### Option 2: Direct Python execution
```bash
python run.py
```

## Features

- **Multi-model Detection**: Supports pose detection, object detection, and action recognition
- **Real-time Processing**: Live video analysis with low latency
- **Modern UI**: Clean PyQt5 interface for intuitive interaction
- **Statistics Tracking**: Real-time progress monitoring and data visualization
- **Video Playback**: Support for both live camera feed and video file input

## Project Structure

```
demo_system/
├── app/                    # Main application modules
│   ├── main_window.py      # Main UI window
│   ├── detection_manager.py # Detection coordination
│   └── video_processor.py   # Video processing logic
├── core/                   # Core detection and processing logic
├── ui/                     # User interface components
├── models/                 # Detection model implementations
│   ├── pose_detector.py    # Pose detection model
│   └── action_recognizer.py # Action recognition model
├── backend_weights/        # Model weights (download from Google Drive)
├── mmaction2/             # MMAction2 framework for action recognition
├── data/                   # Data storage and configuration
├── assets/                 # UI assets and resources
├── run.py                  # Main entry point
└── requirements.txt        # Python dependencies
```

## Notes

- The demo uses pre-trained models that are optimized for geriatric monitoring scenarios
- For best performance, ensure GPU acceleration is properly configured
- The system processes video in real-time, so performance may vary based on hardware capabilities

## Related Documentation

For more information about the full system architecture and other modules, please refer to the main [README](../README.md).

## License

See the main repository license for details.
