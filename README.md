# Digital Biomarker Stability Monitoring for Privacy-Preserving Geriatric Care

**Official repository** for the paper: "Digital biomarker stability monitoring for privacy-preserving geriatric care"

## 📋 Overview

This repository implements an edge-AI framework that enables proactive wellness monitoring for the rapidly expanding geriatric population. The system addresses the critical challenge of tracking functional health continuously and unobtrusively while preserving privacy and independence for older adults.

## 🏗️ System Architecture

The system is organized into main processing modules and training modules:

```
geriatric_monitoring/
├── detection_module/     # Video processing and biomarker extraction
├── analyze_module/       # Data analysis and anomaly detection
├── demo_system/          # System demonstration components
├── demo_web/             # Web-based visualization interface
│
├── deep-person-reid/     # ReID model training (→ detection_module)
├── self_train/           # Detection model fine-tuning via self-training (→ detection_module)
└── Time-Series-Library/   # Time series model training (→ analyze_module)
```

## 🔍 Modules

### 1. Detection Module (`detection_module/`)

**Purpose**: Extract behavioral biomarker sequences from video inputs.

**Key Features**:
- **Human Detection & Tracking**: YOLO-based pose detection and person re-identification (ReID)
- **Face Recognition**: YOLO face detection with recognition capabilities
- **Action Recognition**: Spatial-temporal action detection using MMAction2 framework
- **Biomarker Extraction**: 40 visual biomarkers


**Input**: Video files (`.mov`, `.mp4`, etc.)  
**Output**: CSV files containing timestamped biomarker sequences

**Main Components**:
- `main.py`: Entry point for video processing pipeline
- `video_engine/`: Core video processing engine
- `model/`: Pre-trained models (YOLO, face recognition, action recognition)
- `boxmot/`: Multi-object tracking framework
- `mmaction2/`: Action recognition models

### 2. Analysis Module (`analyze_module/`)

**Purpose**: Analyze biomarker sequences to detect patterns, anomalies, and generate insights.

**Key Features**:
- **Data Preprocessing**: Gaussian smoothing and temporal alignment
- **Factor Analysis**: Dimensionality reduction and pattern extraction
- **Anomaly Detection**: LSTM-based and TimeMixer-based anomaly detection models
- **Time Series Forecasting**: Predictive modeling for future biomarker trends
- **Visualization**: Gantt charts and temporal heatmaps
- **Evaluation Framework**: Comprehensive evaluation metrics for model performance

**Input**: CSV files with biomarker sequences (from detection module)  
**Output**: Analysis results, anomaly alerts, visualizations, and reports

**Main Components**:
- `main_system.py`: Main entry point for analysis pipeline
- `integrate_system/`: Core analysis engine with factor analysis and anomaly detection
- `time_series_model/`: Time series forecasting models
- `filtered_evaluation/`: Evaluation framework for filtered data
- `gcl_evaluation/`: Evaluation framework for GCL (Ground-Truth Comparison) metrics
- `evaluate/`: Human annotation and system evaluation tools

### 3. Demo System (`demo_system/`)

Demonstration components showcasing system capabilities and use cases.

### 4. Demo Web (`demo_web/`)

Web-based visualization interface for displaying monitoring results, including:
- Interactive dashboards
- Temporal heatmaps
- Wellness radar charts
- Alert logs and daily status reports

## 🎓 Training Modules

### 5. Deep Person ReID (`deep-person-reid/`)

**Purpose**: Train person re-identification (ReID) models for use in the detection module.

**Key Features**:
- Person re-identification model training framework
- Supports various ReID architectures and loss functions
- Model evaluation and benchmarking tools

**Output**: Trained ReID model weights used by `detection_module` for person tracking and re-identification.

**Main Components**:
- `torchreid/`: Core ReID framework
- `tools/`: Training and evaluation scripts
- `configs/`: Model configuration files

### 6. Self-Training (`self_train/`)

**Purpose**: Fine-tune detection models (pose detection, action recognition) through self-training techniques.

**Key Features**:
- Self-training pipeline for detection model improvement
- Pseudo-label generation and iterative refinement
- Pose extraction and visualization tools
- Data splitting and preprocessing utilities

**Output**: Fine-tuned detection model weights used by `detection_module` for improved accuracy on domain-specific data.

**Main Components**:
- `train_snh_iterative.py`: Iterative self-training script
- `pseudo_label_generation.py`: Pseudo-label generation
- `evaluate/`: Model evaluation tools
- `preprocessing/`: Data preprocessing utilities

### 7. Time-Series-Library (`Time-Series-Library/`)

**Purpose**: Train time series models (forecasting, anomaly detection, classification) for use in the analysis module.

**Key Features**:
- Comprehensive time series analysis library supporting multiple tasks
- Long- and short-term forecasting, anomaly detection, imputation, and classification
- Multiple model architectures (TimeMixer, TimesNet, iTransformer, LSTM, Transformer-based, etc.)
- Model training, evaluation, and benchmarking tools
- Support for various time series datasets

**Output**: Trained time series model weights (especially anomaly detection models) used by `analyze_module` for detecting anomalies in biomarker sequences.

**Main Components**:
- `models/`: Time series model implementations
- `exp/`: Experiment scripts for different tasks
- `scripts/`: Training scripts for various datasets
- `data_provider/`: Data loading and preprocessing utilities

## 🚀 Quick Start

For detailed installation and usage instructions, please refer to the module-specific documentation:

### Processing Modules
- **[Detection Module](detection_module/README.md)** - Video processing and biomarker extraction
- **[Analysis Module](analyze_module/README.md)** - Data analysis and anomaly detection

### Training Modules
- **[Deep Person ReID](deep-person-reid/README.rst)** - ReID model training
- **[Self-Training](self_train/README.md)** - Detection model fine-tuning
- **[Time-Series-Library](Time-Series-Library/README.md)** - Time series model training


## 🔬 Key Technologies

- **Computer Vision**: YOLO, OpenCV, Dlib
- **Action Recognition**: MMAction2, PoseC3D
- **Person Re-identification**: BotSort, ReID models
- **Time Series Analysis**: LSTM, TimeMixer, TimesNet
- **Anomaly Detection**: Factor Analysis, CUSUM, LSTM-AD
- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Matplotlib, ECharts

## 📁 Project Structure

```
geriatric_monitoring/
├── detection_module/          # Video processing pipeline
│   ├── main.py                # Entry point
│   ├── video_engine/          # Core video processing
│   ├── model/                 # ML models (uses models from training modules)
│   ├── boxmot/                # Tracking framework
│   └── mmaction2/             # Action recognition
│
├── analyze_module/            # Analysis pipeline
│   ├── main_system.py         # Entry point
│   ├── integrate_system/      # Core analysis engine
│   ├── time_series_model/     # Forecasting models (uses models from Time-Series-Library)
│   ├── filtered_evaluation/   # Evaluation framework
│   └── gcl_evaluation/        # GCL evaluation
│
├── demo_system/               # System demonstrations
├── demo_web/                  # Web interface
│
├── deep-person-reid/          # ReID model training
│   ├── torchreid/             # Core ReID framework
│   ├── tools/                 # Training scripts
│   └── configs/               # Model configurations
│
├── self_train/                # Detection model fine-tuning
│   ├── train_snh_iterative.py # Self-training script
│   ├── pseudo_label_generation.py
│   └── evaluate/             # Evaluation tools
│
├── Time-Series-Library/       # Time series model training
│   ├── models/                # Time series model implementations
│   ├── exp/                   # Experiment scripts
│   ├── scripts/               # Training scripts
│   └── data_provider/         # Data loading utilities
│
└── README.md                  # This file
```

## 🔒 Privacy & Ethics

This system is designed for healthcare monitoring applications. Please ensure:
- Proper consent and authorization for video recording
- Compliance with local privacy regulations (HIPAA, GDPR, etc.)
- Secure storage and handling of sensitive health data
- Ethical use of monitoring technologies

## 📝 License

[Specify your license here]

## 🤝 Contributing

[Contributing guidelines]

## 📧 Contact

[Contact information]

---

**Note**: This is an active research and development project. Documentation and features are continuously being improved.
