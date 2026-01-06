#!/usr/bin/env python3
"""Test script to demonstrate CSV existence checking functionality."""

import os
import tempfile
import shutil
import json
from filtered_evaluation.evaluator import FilteredHumanSystemEvaluator

def create_test_data():
    """Create temporary test data to demonstrate the functionality."""
    temp_dir = tempfile.mkdtemp()
    system_output_dir = os.path.join(temp_dir, "system_output")
    human_evaluation_dir = os.path.join(temp_dir, "human_evaluation")
    output_dir = os.path.join(temp_dir, "results")
    
    os.makedirs(system_output_dir, exist_ok=True)
    os.makedirs(human_evaluation_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    test_system_files = [
        "ts_recording_2019_06_22_9_20_am_p_1.csv",
        "ts_recording_2019_06_22_9_20_am_p_2.csv",
        "ts_recording_2019_06_22_9_20_am_p_4.csv",
    ]
    
    for filename in test_system_files:
        filepath = os.path.join(system_output_dir, filename)
        with open(filepath, 'w') as f:
            f.write("seconds,anomaly_scores,variance_curve\n")
            f.write("0,0.1,0.2\n")
            f.write("1,0.3,0.4\n")
    
    test_human_files = ["p_1.csv", "p_2.csv", "p_3.csv", "p_4.csv"]
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
            "p_3": {"train": {"files": [{"filename": "recording_2019_06_22_9_20_am"}]}},
            "p_4": {"train": {"files": [{"filename": "recording_2019_06_22_9_20_am"}]}}
        }
    }
    
    data_splits_path = os.path.join(temp_dir, "data_splits.json")
    with open(data_splits_path, 'w') as f:
        json.dump(data_splits, f)
    
    return temp_dir, system_output_dir, human_evaluation_dir, output_dir, data_splits_path

def test_csv_existence_checking():
    """Test the CSV existence checking functionality."""
    temp_dir, system_output_dir, human_evaluation_dir, output_dir, data_splits_path = create_test_data()
    
    try:
        evaluator = FilteredHumanSystemEvaluator(
            system_output_dir=system_output_dir,
            human_evaluation_filtered_dir=human_evaluation_dir,
            output_dir=output_dir,
            data_splits_path=data_splits_path
        )
        
        evaluator.run_full_evaluation()
        
        if hasattr(evaluator, 'skipped_evaluations'):
            skipped = evaluator.skipped_evaluations
            print(f"Skipped evaluations: {skipped['total_unmatched']}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_csv_existence_checking() 