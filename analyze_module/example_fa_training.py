#!/usr/bin/env python3
"""
Example script demonstrating how to use the Factor Analysis training system with splits.

This script shows how to:
1. Load train/val/test splits from a JSON file
2. Train person-specific Factor Analysis models using only training data
3. Save the trained models for later use in anomaly detection
"""

import os
import sys
import json

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from integrate_system.fa_training import FATrainer
from integrate_system.system_enging import SystemEngine

def load_and_display_splits(splits_file):
    """
    Load and display information about the train/val/test splits.
    
    Args:
        splits_file (str): Path to the splits JSON file
        
    Returns:
        dict: Loaded splits data
    """
    print("="*80)
    print("LOADING TRAIN/VAL/TEST SPLITS")
    print("="*80)
    
    if not os.path.exists(splits_file):
        print(f"✗ Splits file not found: {splits_file}")
        return None
    
    try:
        with open(splits_file, 'r') as f:
            splits_data = json.load(f)
        
        print(f"✓ Loaded splits from: {splits_file}")
        
        # Display metadata
        metadata = splits_data.get('metadata', {})
        print(f"\nMetadata:")
        print(f"  Data directory: {metadata.get('data_directory', 'N/A')}")
        print(f"  Total files: {metadata.get('total_files', 'N/A')}")
        print(f"  Matched files: {metadata.get('matched_files', 'N/A')}")
        print(f"  Unique persons: {metadata.get('unique_persons', 'N/A')}")
        print(f"  Split ratios: {metadata.get('split_ratios', 'N/A')}")
        
        # Display person information
        persons_data = splits_data.get('persons', {})
        print(f"\nPersons with splits: {len(persons_data)}")
        
        # Show summary for each person
        print(f"\nSplit summary by person:")
        total_train_files = 0
        total_val_files = 0
        total_test_files = 0
        
        sorted_persons = sorted(persons_data.keys(), key=lambda x: int(x.split('_')[1]))
        for person_id in sorted_persons:
            person_data = persons_data[person_id]
            train_count = person_data['train']['count']
            val_count = person_data['val']['count']
            test_count = person_data['test']['count']
            
            total_train_files += train_count
            total_val_files += val_count
            total_test_files += test_count
            
            print(f"  {person_id}: {train_count}T + {val_count}V + {test_count}T = {train_count + val_count + test_count} total")
        
        print(f"\nOverall totals:")
        print(f"  Training files: {total_train_files}")
        print(f"  Validation files: {total_val_files}")
        print(f"  Test files: {total_test_files}")
        print(f"  Total: {total_train_files + total_val_files + total_test_files}")
        
        return splits_data
        
    except Exception as e:
        print(f"✗ Error loading splits: {e}")
        return None

def train_fa_models_with_splits():
    """
    Train Factor Analysis models using the train/val/test splits.
    """
    print("\n" + "="*80)
    print("FACTOR ANALYSIS TRAINING WITH SPLITS")
    print("="*80)
    
    # Configuration
    data_directory = "path_to_your_analysis_root/SNH/0_SYSTEM_RESULTS/detection_output/results_v7_alignment_soft/DATASET"
    splits_file = "data_splits.json"
    model_output_directory = "fa_kmo_bartlett"
    min_csv_length = 500
    
    print(f"Data directory: {data_directory}")
    print(f"Splits file: {splits_file}")
    print(f"Model output directory: {model_output_directory}")
    print(f"Minimum CSV length: {min_csv_length}")
    print()
    
    # Check if required files exist
    if not os.path.exists(data_directory):
        print(f"✗ Error: Data directory '{data_directory}' does not exist!")
        return None
    
    if not os.path.exists(splits_file):
        print(f"✗ Error: Splits file '{splits_file}' does not exist!")
        print("Please run create_train_val_test_splits.py first to create the splits.")
        return None
    
    # Create trainer with splits
    trainer = FATrainer(
        data_directory=data_directory,
        model_output_directory=model_output_directory,
        min_csv_length=min_csv_length,
        splits_file=splits_file,
        iterative_drop_experiment=True  # Enable iterative experiment
    )
    
    # Train all models using splits
    print("Starting training process with splits...")
    results = trainer.train_all_models()
    
    return results, model_output_directory

def test_trained_models_example(model_directory, splits_file):
    """
    Example of testing trained models with validation or test data.
    
    Args:
        model_directory (str): Directory containing trained models
        splits_file (str): Path to splits JSON file
    """
    print("\n" + "="*80)
    print("TESTING TRAINED MODELS")
    print("="*80)
    
    # Load splits to find test files
    try:
        with open(splits_file, 'r') as f:
            splits_data = json.load(f)
    except Exception as e:
        print(f"✗ Error loading splits for testing: {e}")
        return
    
    persons_data = splits_data.get('persons', {})
    if not persons_data:
        print("✗ No person data found in splits for testing.")
        return
    
    # Test with the first person's validation files
    first_person = sorted(persons_data.keys(), key=lambda x: int(x.split('_')[1]))[0]
    person_data = persons_data[first_person]
    
    # Use validation files for testing
    val_files = person_data.get('val', {}).get('files', [])
    if not val_files:
        print(f"✗ No validation files found for {first_person}")
        return
    
    # Test with the first validation file
    test_file_info = val_files[0]
    test_file = test_file_info['path']
    
    print(f"Testing with validation file: {test_file_info['filename']}")
    print(f"Person: {first_person}")
    print(f"File path: {test_file}")
    
    # Create SystemEngine and process the test file
    engine = SystemEngine()
    engine.run_name = model_directory  # Set the model directory
    
    try:
        # Process the test file
        results = engine.process_data(test_file)
        print(f"✓ Successfully processed test file with trained models!")
        print(f"  Result type: {type(results)}")
        if hasattr(results, 'shape'):
            print(f"  Result shape: {results.shape}")
        elif hasattr(results, '__len__'):
            print(f"  Result length: {len(results)}")
        
    except Exception as e:
        print(f"✗ Error processing test file: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function demonstrating the complete workflow with splits.
    """
    print("Factor Analysis Training with Train/Val/Test Splits")
    print("This script demonstrates how to train person-specific FA models")
    print("using only training data from the splits.\n")
    
    # Configuration
    splits_file = "data_splits.json"
    
    # Step 1: Load and display splits information
    splits_data = load_and_display_splits(splits_file)
    
    if not splits_data:
        print("\n✗ Failed to load splits. Please run create_train_val_test_splits.py first.")
        return
    
    # Step 2: Train models using splits
    try:
        results, model_dir = train_fa_models_with_splits()
        
        if results:
            successful_models = [pid for pid, result in results.items() if result.get('success', False)]
            failed_models = [pid for pid, result in results.items() if not result.get('success', False)]
            
            print(f"\n{'='*80}")
            print("FINAL TRAINING RESULTS")
            print(f"{'='*80}")
            
            print(f"✓ Training completed!")
            print(f"  Successfully trained: {len(successful_models)} models")
            print(f"  Failed to train: {len(failed_models)} models")
            
            if successful_models:
                print(f"\nSuccessful models:")
                for pid in successful_models:
                    result = results[pid]
                    print(f"  ✓ {pid}: {result['n_factors']} factors, "
                          f"{result['training_samples']} samples, "
                          f"{result['n_training_files']} training files")
            
            if failed_models:
                print(f"\nFailed models:")
                for pid in failed_models:
                    result = results[pid]
                    print(f"  ✗ {pid}: {result.get('error', 'Unknown error')}")
            
            print(f"\nModels saved to: {model_dir}")
            
            # Step 3: Test with a trained model (optional)
            if successful_models:
                print(f"\nTesting trained models...")
                test_trained_models_example(model_dir, splits_file)
            
        else:
            print("✗ No models were trained successfully.")
            
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 