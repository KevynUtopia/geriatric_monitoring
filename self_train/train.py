#!/usr/bin/env python3
"""
Main training script for YOLO11 keypoint detection with self-training support.
"""

import argparse
import logging
from pathlib import Path
import yaml
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.yolo_trainer import YOLO11KeypointTrainer
from src.utils.config_loader import ConfigLoader
from src.training.self_trainer import SelfTrainingMetrics
from ultralytics.utils import LOGGER


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLO11 Keypoint Detection Training")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="Path to dataset configuration file"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "self_train", "evaluate", "predict"],
        default="train",
        help="Training mode"
    )
    
    parser.add_argument(
        "--unlabeled_data", 
        type=str,
        help="Path to unlabeled data for self-training"
    )
    
    parser.add_argument(
        "--model", 
        type=str,
        help="Model name or path to pretrained model"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--device", 
        type=str,
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--project", 
        type=str, 
        default="runs/train",
        help="Project directory for saving results"
    )
    
    parser.add_argument(
        "--name", 
        type=str, 
        default="exp",
        help="Experiment name"
    )
    
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume training from last checkpoint"
    )
    
    parser.add_argument(
        "--wandb", 
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    parser.add_argument(
        "--source", 
        type=str,
        help="Source for prediction (image, video, or directory)"
    )
    
    return parser.parse_args()


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = ConfigLoader.load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config['model']['name'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.device:
        config['device'] = args.device
    if args.wandb:
        config['logging']['wandb']['enabled'] = True
    
    # Setup logging
    log_level = config.get('logging', {}).get('level', 'INFO')
    setup_logging(log_level)
    
    LOGGER.info(f"Starting YOLO11 keypoint detection in {args.mode} mode")
    LOGGER.info(f"Configuration: {args.config}")
    LOGGER.info(f"Dataset: {args.data}")
    
    # Initialize trainer
    trainer = YOLO11KeypointTrainer(config)
    
    try:
        if args.mode == "train":
            # Standard training
            LOGGER.info("Starting standard training")
            results = trainer.train(
                data_config=args.data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                save_dir=args.project
            )
            
            LOGGER.info("Training completed successfully")
            if hasattr(results, 'results_dict'):
                LOGGER.info(f"Final results: {results.results_dict}")
        
        elif args.mode == "self_train":
            # Self-training
            if not args.unlabeled_data:
                raise ValueError("Unlabeled data path required for self-training")
            
            LOGGER.info("Starting self-training")
            
            # Initialize metrics tracking
            metrics_tracker = SelfTrainingMetrics()
            
            results = trainer.self_train(
                labeled_data_config=args.data,
                unlabeled_data_path=args.unlabeled_data,
                max_iterations=config.get('self_training', {}).get('max_iterations', 5)
            )
            
            LOGGER.info("Self-training completed successfully")
            
            # Save self-training summary
            summary = metrics_tracker.get_summary()
            summary_path = Path(args.project) / "self_training_summary.yaml"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w') as f:
                yaml.dump(summary, f, default_flow_style=False)
            
            LOGGER.info(f"Self-training summary saved to: {summary_path}")
        
        elif args.mode == "evaluate":
            # Evaluation
            LOGGER.info("Starting evaluation")
            
            if args.model:
                trainer.load_model(args.model)
            
            results = trainer.evaluate(
                data_config=args.data,
                split="val"
            )
            
            LOGGER.info("Evaluation completed successfully")
            if hasattr(results, 'results_dict'):
                LOGGER.info(f"Evaluation results: {results.results_dict}")
        
        elif args.mode == "predict":
            # Prediction
            if not args.source:
                raise ValueError("Source path required for prediction")
            
            LOGGER.info(f"Making predictions on: {args.source}")
            
            if args.model:
                trainer.load_model(args.model)
            
            results = trainer.predict(
                source=args.source,
                save=True
            )
            
            LOGGER.info(f"Prediction completed. Results saved to: {trainer.save_dir}")
    
    except Exception as e:
        LOGGER.error(f"Training failed with error: {str(e)}")
        raise
    
    finally:
        # Clean up
        trainer.close()
        LOGGER.info("Training session ended")


if __name__ == "__main__":
    main() 
 