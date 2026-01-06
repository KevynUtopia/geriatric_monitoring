# YOLO11 Keypoint Detection with Self-Training

A comprehensive Python and PyTorch repository for keypoint detection using Ultralytics YOLO11 with advanced self-training capabilities. This repository supports both COCO keypoint dataset and custom datasets for human pose estimation.

## Features

- 🚀 **State-of-the-art YOLO11** keypoint detection models
- 🤖 **Self-training** with pseudo-labeling and consistency regularization  
- 📊 **COCO dataset** support with automatic setup
- 🎯 **Custom dataset** integration
- 📈 **Comprehensive evaluation** with detailed metrics and visualizations
- 🔧 **Test-Time Augmentation (TTA)** for robust pseudo-label generation
- 📝 **Curriculum learning** for progressive training
- 🎛️ **Flexible configuration** system
- 📊 **Logging** with TensorBoard and Weights & Biases support

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd self_train

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

#### COCO Dataset
```bash
# Download COCO 2017 dataset (training and validation)
python data_setup.py --dataset coco --data_dir data --coco_year 2017

# Create a sample dataset for quick testing (100 images)
python data_setup.py --dataset coco --data_dir data --sample_size 100
```

#### Custom Dataset
```bash
# Setup custom dataset in COCO format
python data_setup.py --dataset custom --custom_path /path/to/your/dataset --format coco

# Setup custom dataset in YOLO format
python data_setup.py --dataset custom --custom_path /path/to/your/dataset --format yolo
```

### Training

#### Standard Training
```bash
# Train on COCO dataset
python train.py --mode train --data data/coco/dataset.yaml --epochs 100 --batch_size 16

# Train on custom dataset
python train.py --mode train --data data/custom/dataset.yaml --epochs 100 --batch_size 16
```

#### Self-Training
```bash
# Self-training with unlabeled data
python train.py --mode self_train \
    --data data/coco/dataset.yaml \
    --unlabeled_data data/unlabeled_images \
    --epochs 50 \
    --batch_size 16
```

### Evaluation
```bash
# Evaluate trained model
python evaluate.py --model runs/train/exp/weights/best.pt --data data/coco/dataset.yaml --detailed

# Quick evaluation
python train.py --mode evaluate --model runs/train/exp/weights/best.pt --data data/coco/dataset.yaml
```

### Prediction
```bash
# Predict on images or videos
python train.py --mode predict --model runs/train/exp/weights/best.pt --source path/to/images
```

## Configuration

The repository uses YAML configuration files for flexible parameter management. See `config/config.yaml` for the main configuration options.

### Key Configuration Sections

- **Model**: YOLO11 model selection and parameters
- **Training**: Learning rate, epochs, optimizers, etc.
- **Self-training**: Pseudo-labeling thresholds and strategies
- **Dataset**: Data paths and augmentation settings
- **Evaluation**: Metrics and visualization options
- **Logging**: TensorBoard and WandB integration

### Example Configuration Override
```bash
# Override specific parameters
python train.py --mode train --data data/coco/dataset.yaml --epochs 200 --batch_size 32 --model yolo11m-pose.pt
```

## Self-Training Features

### Advanced Pseudo-Labeling
- **Test-Time Augmentation (TTA)**: Ensemble predictions across augmentations
- **Consistency Checking**: Filter pseudo-labels based on prediction consistency
- **Confidence Thresholding**: Dynamic thresholds for high-quality labels

### Curriculum Learning
- **Progressive Training**: Start with easy examples, gradually include harder ones
- **Adaptive Thresholds**: Automatically adjust difficulty based on training progress

### Teacher-Student Framework
- **Exponential Moving Average (EMA)**: Stable teacher model updates
- **Consistency Regularization**: Maintain prediction consistency across augmentations

## Model Zoo

| Model | Size | mAP@0.5 | mAP@0.5:0.95 | Speed (ms) |
|-------|------|---------|--------------|------------|
| YOLO11n-pose | 3.3M | 50.4 | 33.1 | 1.7 |
| YOLO11s-pose | 11.5M | 58.6 | 41.3 | 2.9 |
| YOLO11m-pose | 22.5M | 64.8 | 47.1 | 4.7 |
| YOLO11l-pose | 26.2M | 67.2 | 49.7 | 6.2 |
| YOLO11x-pose | 58.8M | 69.5 | 52.4 | 10.5 |

## Project Structure

```
self_train/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py          # Dataset classes for COCO and custom data
│   ├── models/
│   │   ├── __init__.py
│   │   └── yolo_trainer.py     # YOLO11 trainer with self-training
│   ├── training/
│   │   ├── __init__.py
│   │   └── self_trainer.py     # Advanced self-training utilities
│   └── utils/
│       ├── __init__.py
│       └── config_loader.py    # Configuration management
├── config/
│   └── config.yaml             # Main configuration file
├── data/                       # Dataset directory
├── runs/                       # Training outputs
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation script
├── data_setup.py              # Dataset setup utility
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Usage Examples

### 1. Quick Start with Sample Data
```bash
# Setup sample COCO dataset
python data_setup.py --dataset coco --sample_size 500

# Train on sample data
python train.py --mode train --data data/coco/sample/dataset.yaml --epochs 50

# Evaluate the model
python evaluate.py --model runs/train/exp/weights/best.pt --data data/coco/sample/dataset.yaml --detailed
```

### 2. Self-Training Workflow
```bash
# Initial training on labeled data
python train.py --mode train --data data/labeled/dataset.yaml --epochs 100

# Self-training with unlabeled data
python train.py --mode self_train \
    --data data/labeled/dataset.yaml \
    --unlabeled_data data/unlabeled \
    --model runs/train/exp/weights/best.pt
```

### 3. Custom Dataset Training
```bash
# Setup your custom dataset
python data_setup.py --dataset custom --custom_path /path/to/your/data --format coco

# Train on custom dataset
python train.py --mode train --data data/custom/dataset.yaml --epochs 200
```

## Advanced Features

### Test-Time Augmentation
Enable TTA for more robust predictions:
```python
from src.training.self_trainer import PseudoLabelGenerator

generator = PseudoLabelGenerator(model, config)
pseudo_labels = generator.generate_pseudo_labels(
    unlabeled_images, 
    use_tta=True, 
    use_consistency=True
)
```

### Custom Keypoint Definitions
Define your own keypoint structure:
```yaml
# In dataset.yaml
keypoint_names: ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
kpt_shape: [5, 3]  # 5 keypoints, 3 values each (x, y, visibility)
flip_idx: [0, 2, 1, 4, 3]  # Flip indices for data augmentation
```

### Multi-GPU Training
```bash
# Train on multiple GPUs
python train.py --mode train --data data/coco/dataset.yaml --device 0,1,2,3
```

## Performance Tips

1. **Batch Size**: Use the largest batch size that fits in your GPU memory
2. **Model Selection**: Start with YOLO11s for good speed/accuracy trade-off
3. **Self-Training**: Use high-confidence thresholds (>0.9) for pseudo-labels
4. **Data Augmentation**: Enable mosaic and mixup for better generalization
5. **Learning Rate**: Use cosine scheduling for better convergence

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Low mAP**: Check dataset quality and increase training epochs
3. **Self-training not improving**: Lower pseudo-label threshold or add more unlabeled data

### Debug Mode
```bash
# Run with debug logging
python train.py --mode train --data data/coco/dataset.yaml --config config/debug.yaml
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this repository in your research, please cite:

```bibtex
@misc{yolo11-keypoint-selftraining,
  title={YOLO11 Keypoint Detection with Self-Training},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/yolo11-keypoint-selftraining}}
}
```

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO11 implementation
- [COCO Dataset](https://cocodataset.org/) for the keypoint annotations
- [PyTorch](https://pytorch.org/) for the deep learning framework

## Support

For questions and support, please:
1. Check the [documentation](docs/)
2. Search [existing issues](issues)
3. Create a [new issue](issues/new) if needed 