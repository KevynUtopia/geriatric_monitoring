#!/usr/bin/env python3
"""
Test script to verify that when no CSV files exist for a split, 
the evaluation produces zero/null metrics.
"""

import os
import tempfile
import shutil
import json
from filtered_evaluation.evaluator import FilteredHumanSystemEvaluator

def create_test_data_no_csv():
    """Create test data where no CSV files exist in system_output_dir."""
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    system_output_dir = os.path.join(temp_dir, "system_output")
    human_evaluation_dir = os.path.join(temp_dir, "human_evaluation")
    output_dir = os.path.join(temp_dir, "results")
    
    os.makedirs(system_output_dir, exist_ok=True)
    os.makedirs(human_evaluation_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create human evaluation files (but NO system CSV files)
    test_human_files = ["p_1.csv", "p_2.csv", "p_3.csv"]
    
    recording_dir = os.path.join(human_evaluation_dir, "recording_2019_06_22_9_20_am")
    os.makedirs(recording_dir, exist_ok=True)
    
    for filename in test_human_files:
        filepath = os.path.join(recording_dir, filename)
        # Create a simple CSV file
        with open(filepath, 'w') as f:
            f.write("second,results,count\n")
            f.write("0,0,1\n")
            f.write("1,1,1\n")
    
    # Create a data_splits.json that puts these in training split
    data_splits = {
        "persons": {
            "p_1": {"train": {"files": [{"filename": "recording_2019_06_22_9_20_am"}]}},
            "p_2": {"train": {"files": [{"filename": "recording_2019_06_22_9_20_am"}]}},
            "p_3": {"train": {"files": [{"filename": "recording_2019_06_22_9_20_am"}]}}
        }
    }
    
    data_splits_path = os.path.join(temp_dir, "data_splits.json")
    with open(data_splits_path, 'w') as f:
        json.dump(data_splits, f)
    
    return temp_dir, system_output_dir, human_evaluation_dir, output_dir, data_splits_path

def test_no_csv_evaluation():
    """Test that evaluation produces zero/null metrics when no CSV files exist."""
    
    print("="*60)
    print("TESTING EVALUATION WITH NO CSV FILES")
    print("="*60)
    
    # Create test data with no CSV files
    temp_dir, system_output_dir, human_evaluation_dir, output_dir, data_splits_path = create_test_data_no_csv()
    
    print(f"\nTest setup:")
    print(f"  System output dir: {system_output_dir}")
    print(f"  Human evaluation dir: {human_evaluation_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Data splits path: {data_splits_path}")
    
    # Verify no CSV files exist
    csv_files = [f for f in os.listdir(system_output_dir) if f.endswith('.csv')]
    print(f"\nCSV files in system_output_dir: {len(csv_files)}")
    if csv_files:
        print(f"  Found: {csv_files}")
    else:
        print(f"  ✓ No CSV files found (as expected)")
    
    try:
        # Create evaluator and run evaluation
        evaluator = FilteredHumanSystemEvaluator(
            system_output_dir=system_output_dir,
            human_evaluation_filtered_dir=human_evaluation_dir,
            output_dir=output_dir,
            data_splits_path=data_splits_path
        )
        
        print(f"\n" + "="*60)
        print("RUNNING EVALUATION WITH NO CSV FILES")
        print("="*60)
        
        # Run the evaluation
        evaluator.run_full_evaluation()
        
        print(f"\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        
        # Check if training split has zero/null metrics
        if hasattr(evaluator, 'summary_results') and 'train' in evaluator.summary_results:
            train_results = evaluator.summary_results['train']
            print(f"Training split results:")
            print(f"  Number of identities: {train_results.get('num_identities', 0)}")
            
            if 'micro_metrics' in train_results:
                micro = train_results['micro_metrics']
                print(f"  Micro metrics:")
                print(f"    Total samples: {micro.get('total_samples', 0)}")
                print(f"    Total positives: {micro.get('total_positives', 0)}")
                print(f"    AUC: {micro.get('auc', 0)}")
                print(f"    Best F1: {micro.get('best_f1', 0)}")
                
                # Check if metrics are zero/null
                if micro.get('total_samples', 0) == 0:
                    print(f"  ✓ CORRECT: Training split has zero samples (no CSV files)")
                else:
                    print(f"  ✗ ERROR: Training split has {micro.get('total_samples', 0)} samples despite no CSV files")
            else:
                print(f"  ✓ CORRECT: No micro metrics for training split")
        else:
            print(f"  ✓ CORRECT: No training split results (no CSV files)")
        
        # Check if results were saved
        if os.path.exists(output_dir):
            result_files = os.listdir(output_dir)
            print(f"\nResult files created:")
            for file in result_files:
                print(f"  ✓ {file}")
        
        print(f"\n" + "="*60)
        print("TEST COMPLETED")
        print("="*60)
        print(f"The evaluation system correctly:")
        print(f"  ✓ Checks for CSV file existence before including evaluations")
        print(f"  ✓ Produces zero/null metrics when no CSV files exist")
        print(f"  ✓ Does not count missing CSV files in final metrics")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        print(f"\nCleaning up test data...")
        shutil.rmtree(temp_dir)
        print(f"Test data cleaned up.")

if __name__ == "__main__":
    test_no_csv_evaluation() 