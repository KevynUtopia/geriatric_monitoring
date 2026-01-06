"""
Configuration loading and validation utilities.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate configuration files."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Validate configuration
        validated_config = ConfigLoader.validate_config(config)
        
        logger.info(f"Loaded configuration from {config_path}")
        return validated_config
    
    @staticmethod
    def load_hydra_config(config_path: str, overrides: Optional[list] = None) -> DictConfig:
        """
        Load configuration using Hydra/OmegaConf.
        
        Args:
            config_path: Path to configuration file
            overrides: List of configuration overrides
            
        Returns:
            OmegaConf configuration
        """
        config = OmegaConf.load(config_path)
        
        if overrides:
            override_config = OmegaConf.from_dotlist(overrides)
            config = OmegaConf.merge(config, override_config)
        
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and set default values for configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration dictionary
        """
        # Set default values
        defaults = {
            'model': {
                'name': 'yolo11n-pose.pt',
                'input_size': 640,
                'confidence_threshold': 0.25,
                'iou_threshold': 0.7,
                'max_detections': 300
            },
            'training': {
                'epochs': 300,
                'batch_size': 16,
                'learning_rate': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'optimizer': 'SGD',
                'scheduler': 'cosine',
                'patience': 50,
                'save_period': 10,
                'save_best': True
            },
            'self_training': {
                'enabled': False,
                'pseudo_label_threshold': 0.9,
                'max_iterations': 5,
                'unlabeled_data_ratio': 0.3,
                'teacher_student_ema': 0.999,
                'consistency_weight': 1.0,
                'ramp_up_epochs': 30
            },
            'evaluation': {
                'metrics': ['precision', 'recall', 'mAP50', 'mAP50-95'],
                'save_plots': True,
                'save_json': True,
                'conf_threshold': 0.001,
                'iou_threshold': 0.6
            },
            'logging': {
                'wandb': {
                    'enabled': False,
                    'project': 'yolo11-keypoint-detection',
                    'entity': None
                },
                'tensorboard': {
                    'enabled': True,
                    'log_dir': 'runs'
                },
                'level': 'INFO'
            },
            'device': 'auto',
            'workers': 8
        }
        
        # Merge with defaults
        validated_config = _deep_merge(defaults, config)
        
        # Validate specific constraints
        ConfigLoader._validate_constraints(validated_config)
        
        return validated_config
    
    @staticmethod
    def _validate_constraints(config: Dict[str, Any]):
        """Validate configuration constraints."""
        # Model constraints
        valid_models = [
            'yolo11n-pose.pt', 'yolo11s-pose.pt', 'yolo11m-pose.pt', 
            'yolo11l-pose.pt', 'yolo11x-pose.pt'
        ]
        
        model_name = config['model']['name']
        if not any(model_name.startswith(vm.split('.')[0]) for vm in valid_models):
            logger.warning(f"Model {model_name} might not be a valid YOLO11 pose model")
        
        # Training constraints
        if config['training']['batch_size'] <= 0:
            raise ValueError("Batch size must be positive")
        
        if config['training']['epochs'] <= 0:
            raise ValueError("Epochs must be positive")
        
        if config['training']['learning_rate'] <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Self-training constraints
        if config['self_training']['enabled']:
            if not (0 < config['self_training']['pseudo_label_threshold'] <= 1):
                raise ValueError("Pseudo label threshold must be between 0 and 1")
            
            if config['self_training']['max_iterations'] <= 0:
                raise ValueError("Max iterations must be positive")
        
        logger.info("Configuration validation passed")
    
    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str):
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            save_path: Path to save configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            if save_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif save_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {save_path.suffix}")
        
        logger.info(f"Saved configuration to {save_path}")


def _deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """Recursively merge two dictionaries."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def create_dataset_config(
    dataset_path: str,
    train_split: str = "train",
    val_split: str = "val",
    num_classes: int = 1,
    class_names: Optional[list] = None,
    keypoint_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Create a dataset configuration for YOLO.
    
    Args:
        dataset_path: Path to dataset root
        train_split: Training split name
        val_split: Validation split name
        num_classes: Number of classes
        class_names: List of class names
        keypoint_names: List of keypoint names
        
    Returns:
        Dataset configuration dictionary
    """
    if class_names is None:
        class_names = ['person']
    
    if keypoint_names is None:
        # Default COCO keypoints
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    config = {
        'path': str(Path(dataset_path).absolute()),
        'train': train_split,
        'val': val_split,
        'nc': num_classes,
        'names': class_names,
        'kpt_shape': [len(keypoint_names), 3],  # [num_keypoints, 3] for x,y,visibility
        'flip_idx': [],  # Define flip indices for data augmentation
        'keypoint_names': keypoint_names
    }
    
    # Define flip indices for human pose (COCO format)
    if len(keypoint_names) == 17:
        config['flip_idx'] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    
    return config 