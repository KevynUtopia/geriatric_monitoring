#!/usr/bin/env python3
"""Test script to verify that when no CSV files exist for a split, the evaluation produces zero/null metrics."""

import os
import tempfile
import shutil
import json
from filtered_evaluation.evaluator import FilteredHumanSystemEvaluator

def create_test_data_no_csv():
    """Create test data where no CSV files exist in system_output_dir."""
    temp_dir = tempfile.mkdtemp()
    system_output_dir = os.path.join(temp_dir, "system_output")
    human_evaluation_dir = os.path.join(temp_dir, "human_evaluation")
    output_dir = os.path.join(temp_dir, "results")
    
    os.makedirs(system_output_dir, exist_ok=True)
    os.makedirs(human_evaluation_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    test_human_files = ["p_1.csv", "p_2.csv", "p_3.csv"]
    recording_dir = os.path.join(human_evaluation_dir, "recording_2019_06_22_9_20_am")
    os.makedirs(recording_dir, exist_ok=True)
    
    for filename in test_human_files:
        filepath = os.path.join(recording_dir, filename)
        with open(filepath, 'w') as f:
            f.write("second,results,count\n")
            f.write("0,0,1\n")
            f.write("1,1,1\n")
    
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
    temp_dir, system_output_dir, human_evaluation_dir, output_dir, data_splits_path = create_test_data_no_csv()
    
    try:
        evaluator = FilteredHumanSystemEvaluator(
            system_output_dir=system_output_dir,
            human_evaluation_filtered_dir=human_evaluation_dir,
            output_dir=output_dir,
            data_splits_path=data_splits_path
        )
        
        evaluator.run_full_evaluation()
        
        if hasattr(evaluator, 'summary_results') and 'train' in evaluator.summary_results:
            train_results = evaluator.summary_results['train']
            if 'micro_metrics' in train_results:
                micro = train_results['micro_metrics']
                if micro.get('total_samples', 0) == 0:
                    print("Test passed: Training split has zero samples")
                else:
                    print(f"Test failed: Training split has {micro.get('total_samples', 0)} samples")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_no_csv_evaluation() 