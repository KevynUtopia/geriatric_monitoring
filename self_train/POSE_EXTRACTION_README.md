# Video Pose Extraction System

This system processes videos to extract frames and generate pose labels in YOLO format using YOLO11 pose detection models.

## Features

- **Batch Video Processing**: Process multiple videos organized in day/camera directory structure
- **YOLO Pose Detection**: Uses YOLO11 pose detection models for accurate human pose estimation
- **YOLO Format Output**: Generates labels compatible with YOLO training format
- **Resume Capability**: Can resume processing from where it left off

- **Robust Error Handling**: Continues processing even if some frames fail

## Output Format

The system generates:
- **Images**: Extracted frames as `.jpg` files in a single `images/` directory
- **Labels**: Pose annotations in YOLO format as `.txt` files in a single `labels/` directory  
- **Tracking**: JSON files tracking processing status in original directory structure
- **Filenames**: Combined format `day_camera_video_framenum` for unique identification

### YOLO Label Format
Each label file contains lines in the format:
```
class_id center_x center_y width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ... kp17_x kp17_y kp17_v
```

Where:
- `class_id`: Always 0 (person class)
- `center_x, center_y, width, height`: Normalized bounding box coordinates
- `kpN_x, kpN_y, kpN_v`: Normalized keypoint coordinates and visibility (0=not visible, 2=visible)

## Installation

### Requirements
```bash
pip install torch ultralytics opencv-python numpy pytz
```

### File Structure
```
your_project/
├── pseudo_label_generation.py    # Main processing script
├── video_engine.py               # Video processing engine
├── run_pose_extraction.sh        # Bash execution script
├── example_usage.py              # Example usage script
└── POSE_EXTRACTION_README.md     # This file
```

## Usage

### Method 1: Bash Script (Recommended)

```bash
# Make script executable
chmod +x run_pose_extraction.sh

# Basic usage
./run_pose_extraction.sh -i /path/to/videos -o /path/to/output

# With all options
./run_pose_extraction.sh \
    -i /path/to/input/videos \
    -o /path/to/output \
    -m /path/to/model.pt \
    -c 0.95 \
    -f 5 \
    -r
```

**Options:**
- `-i, --input_path`: Path to input video directory (required)
- `-o, --output_path`: Path to output directory (required)  
- `-m, --model_path`: Path to YOLO pose model (optional, defaults to yolo11l-pose.pt)
- `-r, --resume`: Resume processing from where it left off
- `-c, --confidence`: Confidence threshold for pose detection (default: 0.95)
- `-f, --frame_interval`: Frame interval to process (default: 1)
- `-s, --seed`: Random seed (default: 42)

### Method 2: Direct Python

```bash
python3 pseudo_label_generation.py \
    --input_path /path/to/videos \
    --out_path /path/to/output \
    --confidence_threshold 0.95 \
    --frame_interval 5 \
    --resume
```

### Method 3: Programmatic Usage

```python
from pseudo_label_generation import main

class Args:
    def __init__(self):
        self.input_path = "/path/to/videos"
        self.out_path = "/path/to/output"
        self.model_path = "yolo11l-pose.pt"
        self.confidence_threshold = 0.95
        self.frame_interval = 5
        self.resume = True
        self.clip_len = 16
        self.seed = 42
        self.run_name = None

args = Args()
main(args)
```

## Input Directory Structure

Your input videos should be organized as:
```
input_videos/
├── day1/
│   ├── camera1/
│   │   ├── video1.mp4
│   │   └── video2.mp4
│   └── camera2/
│       └── video3.mp4
└── day2/
    └── camera1/
        └── video4.mp4
```

## Output Directory Structure

The system generates:
```
output/
├── images/
│   ├── day1_camera1_video1_000000000000.jpg
│   ├── day1_camera1_video1_000000000001.jpg
│   ├── day1_camera2_video3_000000000000.jpg
│   └── day2_camera1_video4_000000000000.jpg
├── labels/
│   ├── day1_camera1_video1_000000000000.txt
│   ├── day1_camera1_video1_000000000001.txt
│   ├── day1_camera2_video3_000000000000.txt
│   └── day2_camera1_video4_000000000000.txt
└── day1/
    ├── camera1/
    │   └── list.json  # Processing status
    ├── camera2/
    │   └── list.json
    └── day2/
        └── camera1/
            └── list.json
```

## Examples

### Example 1: Create Sample Structure
```bash
python3 example_usage.py --create_sample
```

### Example 2: Process with Resume
```bash
./run_pose_extraction.sh \
    -i /data/videos \
    -o /data/processed \
    -r  # Resume from previous run
```

### Example 3: Process Every 10th Frame
```bash
./run_pose_extraction.sh \
    -i /data/videos \
    -o /data/processed \
    -f 10  # Process every 10th frame
```

### Example 4: Lower Confidence Threshold
```bash
./run_pose_extraction.sh \
    -i /data/videos \
    -o /data/processed \
    -c 0.7  # Lower confidence threshold
```

## Performance Tips

1. **GPU Usage**: The system automatically uses GPU if available
2. **Frame Interval**: Use `--frame_interval 5` to process every 5th frame for faster processing
3. **Confidence Threshold**: Higher values (0.9+) give fewer but more accurate detections
4. **Resume**: Use `--resume` to continue interrupted processing sessions
5. **Memory**: For very large videos, consider processing in smaller batches

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Video codec issues**: Ensure videos are in supported formats (mp4, avi, mov, mkv)
3. **Permission errors**: Check write permissions for output directory
4. **Missing dependencies**: Run the bash script which checks dependencies

### Checking System Status
```bash
# Check if all dependencies are installed
python3 -c "import torch, cv2, ultralytics; print('All dependencies OK')"

# Check GPU availability
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Model Information

The system uses YOLO11 pose detection models:
- **Default**: `yolo11l-pose.pt` (large model, good accuracy)
- **Alternatives**: `yolo11n-pose.pt` (nano), `yolo11s-pose.pt` (small), `yolo11m-pose.pt` (medium), `yolo11x-pose.pt` (extra large)

Models are automatically downloaded from Ultralytics if not found locally.

## Integration with Training

The generated labels can be directly used for YOLO training:

```python
# Example dataset.yaml for training  
path: /path/to/processed/data  # Point to your output directory
train: images  # All images are in single directory
val: images    # Split can be done by filename patterns

kpt_shape: [17, 3]  # 17 keypoints, 3 values (x, y, visibility)
names: ['person']

# Note: Filenames follow pattern day_camera_video_framenum.jpg/txt
# You can split train/val by day, camera, or video patterns
``` 