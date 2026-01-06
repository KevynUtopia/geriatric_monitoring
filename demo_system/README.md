# AI Detection Template

A template project inspired by Good-GYM with similar UI but different backend computation capabilities.

## Features

- **Real-time Detection** - Multiple detection models working in parallel
- **Modern UI** - Clean PyQt5 interface similar to Good-GYM
- **Multi-Model Support** - Skeleton code for various detection models
- **Video Processing** - Real-time video analysis and display
- **Statistics Tracking** - Progress monitoring and data visualization
- **Extensible Architecture** - Easy to add new detection models

## Project Structure

```
├── app/                    # Main application modules
├── core/                   # Core detection and processing logic
├── ui/                     # User interface components
├── models/                 # Detection model implementations
├── data/                   # Data storage and configuration
├── assets/                 # UI assets and resources
└── requirements.txt        # Python dependencies
```

## Backend Models (Skeleton)

The template includes skeleton code for multiple detection models:

1. **Pose Detection** - Human pose estimation
2. **Object Detection** - General object recognition
3. **Action Recognition** - Human activity classification
4. **Custom Models** - Placeholder for your specific models

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python run.py
   ```

## Development

This template provides a solid foundation for building AI detection applications. The backend computation can be customized by implementing the skeleton model classes in the `models/` directory.

## License

MIT License - Feel free to use and modify for your projects.

