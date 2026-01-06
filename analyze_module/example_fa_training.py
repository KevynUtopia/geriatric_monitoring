#!/usr/bin/env python3
"""Example script demonstrating Factor Analysis training with splits."""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from integrate_system.fa_training import FATrainer
from integrate_system.system_enging import SystemEngine

def load_and_display_splits(splits_file):
    """Load and display information about the train/val/test splits."""
    if not os.path.exists(splits_file):
        print(f"Splits file not found: {splits_file}")
        return None
    
    try:
        with open(splits_file, 'r') as f:
            splits_data = json.load(f)
        return splits_data
    except Exception as e:
        print(f"Error loading splits: {e}")
        return None

def train_fa_models_with_splits():
    """Train Factor Analysis models using the train/val/test splits."""
    data_directory = "path_to_your_analysis_root/SNH/0_SYSTEM_RESULTS/detection_output/results_alignment_soft/DATASET"
    splits_file = "data_splits.json"
    model_output_directory = "fa_kmo_bartlett"
    min_csv_length = 500
    
    if not os.path.exists(data_directory):
        print(f"Error: Data directory '{data_directory}' does not exist!")
        return None
    
    if not os.path.exists(splits_file):
        print(f"Error: Splits file '{splits_file}' does not exist!")
        return None
    
    trainer = FATrainer(
        data_directory=data_directory,
        model_output_directory=model_output_directory,
        min_csv_length=min_csv_length,
        splits_file=splits_file,
        iterative_drop_experiment=True
    )
    
    results = trainer.train_all_models()
    return results, model_output_directory

def test_trained_models_example(model_directory, splits_file):
    """Example of testing trained models with validation or test data."""
    try:
        with open(splits_file, 'r') as f:
            splits_data = json.load(f)
    except Exception as e:
        print(f"Error loading splits for testing: {e}")
        return
    
    persons_data = splits_data.get('persons', {})
    if not persons_data:
        return
    
    first_person = sorted(persons_data.keys(), key=lambda x: int(x.split('_')[1]))[0]
    person_data = persons_data[first_person]
    val_files = person_data.get('val', {}).get('files', [])
    
    if not val_files:
        return
    
    test_file_info = val_files[0]
    test_file = test_file_info['path']
    
    engine = SystemEngine()
    engine.run_name = model_directory
    
    try:
        results = engine.process_data(test_file)
    except Exception as e:
        print(f"Error processing test file: {e}")

def main():
    """Main function demonstrating the complete workflow with splits."""
    splits_file = "data_splits.json"
    splits_data = load_and_display_splits(splits_file)
    
    if not splits_data:
        return
    
    try:
        results, model_dir = train_fa_models_with_splits()
        
        if results:
            successful_models = [pid for pid, result in results.items() if result.get('success', False)]
            failed_models = [pid for pid, result in results.items() if not result.get('success', False)]
            
            print(f"Training completed: {len(successful_models)} successful, {len(failed_models)} failed")
            print(f"Models saved to: {model_dir}")
            
            if successful_models:
                test_trained_models_example(model_dir, splits_file)
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main() 