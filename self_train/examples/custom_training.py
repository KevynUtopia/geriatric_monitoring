#!/usr/bin/env python3
"""
Example script demonstrating custom training workflows with YOLO11 keypoint detection.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.yolo_trainer import YOLO11KeypointTrainer
from src.utils.config_loader import ConfigLoader
from src.training.self_trainer import PseudoLabelGenerator, CurriculumLearning, SelfTrainingMetrics


def example_standard_training():
    """Example of standard training workflow."""
    print("=== Standard Training Example ===")
    
    # Load configuration
    config = ConfigLoader.load_config("config/config.yaml")
    
    # Initialize trainer
    trainer = YOLO11KeypointTrainer(config)
    
    # Train the model
    results = trainer.train(
        data_config="data/coco/dataset.yaml",
        epochs=50,
        batch_size=16,
        imgsz=640
    )
    
    print("Training completed!")
    print(f"Results: {results.results_dict if hasattr(results, 'results_dict') else 'No results dict'}")
    
    # Clean up
    trainer.close()


def example_self_training():
    """Example of self-training workflow."""
    print("=== Self-Training Example ===")
    
    # Load configuration with self-training enabled
    config = ConfigLoader.load_config("config/config.yaml")
    config['self_training']['enabled'] = True
    config['self_training']['pseudo_label_threshold'] = 0.85
    config['self_training']['max_iterations'] = 3
    
    # Initialize trainer
    trainer = YOLO11KeypointTrainer(config)
    
    # Initialize metrics tracking
    metrics_tracker = SelfTrainingMetrics()
    
    # Perform self-training
    results = trainer.self_train(
        labeled_data_config="data/coco/dataset.yaml",
        unlabeled_data_path="data/unlabeled_images",
        max_iterations=3
    )
    
    print("Self-training completed!")
    
    # Get training summary
    summary = metrics_tracker.get_summary()
    print(f"Training summary: {summary}")
    
    # Clean up
    trainer.close()


def example_advanced_pseudo_labeling():
    """Example of advanced pseudo-label generation."""
    print("=== Advanced Pseudo-Labeling Example ===")
    
    # Load configuration
    config = ConfigLoader.load_config("config/config.yaml")
    
    # Initialize trainer and load a pre-trained model
    trainer = YOLO11KeypointTrainer(config)
    # trainer.load_model("path/to/pretrained/model.pt")  # Uncomment to load specific model
    
    # Initialize pseudo-label generator
    generator = PseudoLabelGenerator(trainer.model, config)
    
    # Get list of unlabeled images
    unlabeled_images = [
        "data/unlabeled/image1.jpg",
        "data/unlabeled/image2.jpg",
        "data/unlabeled/image3.jpg",
    ]
    
    # Generate pseudo labels with different strategies
    print("Generating pseudo labels with TTA and consistency...")
    pseudo_labels = generator.generate_pseudo_labels(
        unlabeled_images,
        use_tta=True,
        use_consistency=True
    )
    
    print(f"Generated {len(pseudo_labels)} high-quality pseudo labels")
    
    # Apply curriculum learning
    curriculum = CurriculumLearning(config)
    filtered_labels = curriculum.filter_pseudo_labels_by_difficulty(
        pseudo_labels, 
        iteration=1, 
        max_iterations=5
    )
    
    print(f"After curriculum filtering: {len(filtered_labels)} labels")
    
    # Clean up
    trainer.close()


def example_custom_evaluation():
    """Example of custom model evaluation."""
    print("=== Custom Evaluation Example ===")
    
    # Load configuration
    config = ConfigLoader.load_config("config/config.yaml")
    
    # Initialize trainer
    trainer = YOLO11KeypointTrainer(config)
    
    # Load a trained model
    # trainer.load_model("runs/train/exp/weights/best.pt")  # Uncomment to load specific model
    
    # Evaluate on validation set
    results = trainer.evaluate(
        data_config="data/coco/dataset.yaml",
        split="val"
    )
    
    print("Evaluation completed!")
    print(f"mAP@0.5: {getattr(results, 'box', {}).get('map50', 'N/A')}")
    print(f"mAP@0.5:0.95: {getattr(results, 'box', {}).get('map', 'N/A')}")
    
    # Clean up
    trainer.close()


def example_prediction_workflow():
    """Example of prediction workflow."""
    print("=== Prediction Workflow Example ===")
    
    # Load configuration
    config = ConfigLoader.load_config("config/config.yaml")
    
    # Initialize trainer
    trainer = YOLO11KeypointTrainer(config)
    
    # Load a trained model
    # trainer.load_model("runs/train/exp/weights/best.pt")  # Uncomment to load specific model
    
    # Make predictions on images
    results = trainer.predict(
        source="data/test_images",  # Can be image, directory, or video
        save=True,
        conf=0.25,
        iou=0.7
    )
    
    print(f"Prediction completed! Results saved to: {trainer.save_dir}")
    
    # Process results
    for result in results:
        if result.keypoints is not None:
            print(f"Detected {len(result.keypoints)} poses in {result.path}")
    
    # Clean up
    trainer.close()


def example_multi_model_comparison():
    """Example of comparing different YOLO11 models."""
    print("=== Multi-Model Comparison Example ===")
    
    models = ['yolo11n-pose.pt', 'yolo11s-pose.pt', 'yolo11m-pose.pt']
    results_comparison = {}
    
    for model_name in models:
        print(f"Evaluating {model_name}...")
        
        # Load configuration
        config = ConfigLoader.load_config("config/config.yaml")
        config['model']['name'] = model_name
        
        # Initialize trainer
        trainer = YOLO11KeypointTrainer(config)
        
        # Quick training (for demo purposes)
        results = trainer.train(
            data_config="data/coco/sample/dataset.yaml",  # Use sample dataset
            epochs=5,  # Short training
            batch_size=8
        )
        
        # Store results
        if hasattr(results, 'results_dict'):
            results_comparison[model_name] = results.results_dict
        
        trainer.close()
    
    # Print comparison
    print("\n=== Model Comparison Results ===")
    for model, metrics in results_comparison.items():
        print(f"{model}: mAP = {metrics.get('metrics/mAP50(B)', 'N/A')}")


def main():
    """Run all examples."""
    print("YOLO11 Keypoint Detection Examples")
    print("=" * 50)
    
    # Run examples (comment out as needed)
    try:
        example_standard_training()
    except Exception as e:
        print(f"Standard training example failed: {e}")
    
    print("\n" + "=" * 50)
    
    try:
        example_self_training()
    except Exception as e:
        print(f"Self-training example failed: {e}")
    
    print("\n" + "=" * 50)
    
    try:
        example_advanced_pseudo_labeling()
    except Exception as e:
        print(f"Advanced pseudo-labeling example failed: {e}")
    
    print("\n" + "=" * 50)
    
    try:
        example_custom_evaluation()
    except Exception as e:
        print(f"Custom evaluation example failed: {e}")
    
    print("\n" + "=" * 50)
    
    try:
        example_prediction_workflow()
    except Exception as e:
        print(f"Prediction workflow example failed: {e}")
    
    print("\n" + "=" * 50)
    
    # Uncomment to run model comparison (takes longer)
    # try:
    #     example_multi_model_comparison()
    # except Exception as e:
    #     print(f"Multi-model comparison example failed: {e}")


if __name__ == "__main__":
    main() 