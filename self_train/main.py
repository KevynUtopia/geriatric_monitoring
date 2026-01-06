#!/usr/bin/env python3
"""
YOLO11 Keypoint Detection with Self-Training
Main entry point for the training system.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="YOLO11 Keypoint Detection with Self-Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training
  python main.py train --data data/coco/dataset.yaml --epochs 100

  # Self-training
  python main.py self-train --data data/coco/dataset.yaml --unlabeled data/unlabeled

  # Evaluation
  python main.py evaluate --model runs/train/exp/weights/best.pt --data data/coco/dataset.yaml

  # Data setup
  python main.py setup-data --dataset coco --data_dir data

  # Prediction
  python main.py predict --model runs/train/exp/weights/best.pt --source images/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train YOLO11 model')
    train_parser.add_argument('--data', type=str, required=True, help='Dataset config path')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--model', type=str, default='yolo11n-pose.pt', help='Model name')
    train_parser.add_argument('--device', type=str, default='auto', help='Device to use')
    train_parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    
    # Self-training command
    self_train_parser = subparsers.add_parser('self-train', help='Self-training with pseudo labels')
    self_train_parser.add_argument('--data', type=str, required=True, help='Labeled dataset config')
    self_train_parser.add_argument('--unlabeled', type=str, required=True, help='Unlabeled data path')
    self_train_parser.add_argument('--iterations', type=int, default=5, help='Self-training iterations')
    self_train_parser.add_argument('--model', type=str, default='yolo11n-pose.pt', help='Model name')
    self_train_parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model', type=str, required=True, help='Model path')
    eval_parser.add_argument('--data', type=str, required=True, help='Dataset config path')
    eval_parser.add_argument('--split', type=str, default='val', help='Dataset split')
    eval_parser.add_argument('--detailed', action='store_true', help='Generate detailed analysis')
    eval_parser.add_argument('--save-dir', type=str, default='runs/eval', help='Save directory')
    
    # Data setup command
    setup_parser = subparsers.add_parser('setup-data', help='Setup datasets')
    setup_parser.add_argument('--dataset', type=str, choices=['coco', 'custom'], required=True, help='Dataset type')
    setup_parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    setup_parser.add_argument('--sample-size', type=int, help='Create sample dataset')
    setup_parser.add_argument('--custom-path', type=str, help='Custom dataset path')
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', type=str, required=True, help='Model path')
    predict_parser.add_argument('--source', type=str, required=True, help='Source path')
    predict_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    predict_parser.add_argument('--save', action='store_true', help='Save results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Import and run the appropriate script
    if args.command == 'train':
        from train import main as train_main
        sys.argv = [
            'train.py', '--mode', 'train',
            '--data', args.data,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--model', args.model,
            '--device', args.device,
            '--config', args.config
        ]
        train_main()
    
    elif args.command == 'self-train':
        from train import main as train_main
        sys.argv = [
            'train.py', '--mode', 'self_train',
            '--data', args.data,
            '--unlabeled_data', args.unlabeled,
            '--model', args.model,
            '--config', args.config
        ]
        train_main()
    
    elif args.command == 'evaluate':
        from evaluate import main as eval_main
        sys.argv = [
            'evaluate.py',
            '--model', args.model,
            '--data', args.data,
            '--split', args.split,
            '--save_dir', args.save_dir
        ]
        if args.detailed:
            sys.argv.append('--detailed')
        eval_main()
    
    elif args.command == 'setup-data':
        from data_setup import main as setup_main
        sys.argv = [
            'data_setup.py',
            '--dataset', args.dataset,
            '--data_dir', args.data_dir
        ]
        if args.sample_size:
            sys.argv.extend(['--sample_size', str(args.sample_size)])
        if args.custom_path:
            sys.argv.extend(['--custom_path', args.custom_path])
        setup_main()
    
    elif args.command == 'predict':
        from train import main as train_main
        sys.argv = [
            'train.py', '--mode', 'predict',
            '--model', args.model,
            '--source', args.source
        ]
        train_main()


if __name__ == "__main__":
    main()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
