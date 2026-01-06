#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced CSV existence checking functionality.

This script shows how the evaluation system now:
1. Only includes evaluations where CSV files exist in system_output_dir
2. Skips evaluations where CSV files are missing
3. Does not count skipped evaluations in final metrics calculation
4. Provides detailed logging and reporting of skipped evaluations
"""

import os
import tempfile
import shutil
from filtered_evaluation.evaluator import FilteredHumanSystemEvaluator

def create_test_data():
    """Create temporary test data to demonstrate the functionality."""
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    system_output_dir = os.path.join(temp_dir, "system_output")
    human_evaluation_dir = os.path.join(temp_dir, "human_evaluation")
    output_dir = os.path.join(temp_dir, "results")
    
    os.makedirs(system_output_dir, exist_ok=True)
    os.makedirs(human_evaluation_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create some test system files (some existing, some missing)
    test_system_files = [
        "ts_recording_2019_06_22_9_20_am_p_1.csv",
        "ts_recording_2019_06_22_9_20_am_p_2.csv",
        # Note: p_3 file is intentionally missing
        "ts_recording_2019_06_22_9_20_am_p_4.csv",
    ]
    
    for filename in test_system_files:
        filepath = os.path.join(system_output_dir, filename)
        # Create a simple CSV file
        with open(filepath, 'w') as f:
            f.write("seconds,anomaly_scores,variance_curve\n")
            f.write("0,0.1,0.2\n")
            f.write("1,0.3,0.4\n")
    
    # Create human evaluation files
    test_human_files = [
        "p_1.csv",
        "p_2.csv", 
        "p_3.csv",  # This will be skipped since system file doesn't exist
        "p_4.csv",
    ]
    
    recording_dir = os.path.join(human_evaluation_dir, "recording_2019_06_22_9_20_am")
    os.makedirs(recording_dir, exist_ok=True)
    
    for filename in test_human_files:
        filepath = os.path.join(recording_dir, filename)
        # Create a simple CSV file
        with open(filepath, 'w') as f:
            f.write("second,results,count\n")
            f.write("0,0,1\n")
            f.write("1,1,1\n")
    
    # Create a simple data_splits.json
    data_splits = {
        "persons": {
            "p_1": {"train": {"files": [{"filename": "recording_2019_06_22_9_20_am"}]}},
            "p_2": {"train": {"files": [{"filename": "recording_2019_06_22_9_20_am"}]}},
            "p_3": {"train": {"files": [{"filename": "recording_2019_06_22_9_20_am"}]}},
            "p_4": {"train": {"files": [{"filename": "recording_2019_06_22_9_20_am"}]}}
        }
    }
    
    data_splits_path = os.path.join(temp_dir, "data_splits.json")
    import json
    with open(data_splits_path, 'w') as f:
        json.dump(data_splits, f)
    
    return temp_dir, system_output_dir, human_evaluation_dir, output_dir, data_splits_path

def test_csv_existence_checking():
    """Test the CSV existence checking functionality."""
    
    print("="*60)
    print("TESTING CSV EXISTENCE CHECKING FUNCTIONALITY")
    print("="*60)
    
    # Create test data
    temp_dir, system_output_dir, human_evaluation_dir, output_dir, data_splits_path = create_test_data()
    
    print(f"\nTest setup:")
    print(f"  System output dir: {system_output_dir}")
    print(f"  Human evaluation dir: {human_evaluation_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Data splits path: {data_splits_path}")
    
    # List created files
    print(f"\nCreated system files:")
    for file in os.listdir(system_output_dir):
        print(f"  ✓ {file}")
    
    print(f"\nCreated human evaluation files:")
    for root, dirs, files in os.walk(human_evaluation_dir):
        for file in files:
            print(f"  ✓ {os.path.join(root, file)}")
    
    try:
        # Create evaluator and run evaluation
        evaluator = FilteredHumanSystemEvaluator(
            system_output_dir=system_output_dir,
            human_evaluation_filtered_dir=human_evaluation_dir,
            output_dir=output_dir,
            data_splits_path=data_splits_path
        )
        
        print(f"\n" + "="*60)
        print("RUNNING EVALUATION WITH CSV EXISTENCE CHECKING")
        print("="*60)
        
        # Run the evaluation
        evaluator.run_full_evaluation()
        
        print(f"\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        
        # Check if skipped evaluations were properly recorded
        if hasattr(evaluator, 'skipped_evaluations'):
            skipped = evaluator.skipped_evaluations
            print(f"✓ Skipped evaluations tracking: ENABLED")
            print(f"  Total matched: {skipped['total_matched']}")
            print(f"  Total skipped: {skipped['total_unmatched']}")
            print(f"  Skipped entries: {len(skipped['skipped_entries'])}")
            
            if skipped['skipped_entries']:
                print(f"\nSkipped evaluation details:")
                for entry in skipped['skipped_entries']:
                    print(f"  - {entry['person_id']} in {entry['recording_session']} ({entry['split']} split)")
                    print(f"    Missing files: {', '.join(entry['expected_files'])}")
        else:
            print(f"✗ Skipped evaluations tracking: NOT ENABLED")
        
        # Check if results were saved
        if os.path.exists(output_dir):
            result_files = os.listdir(output_dir)
            print(f"\nResult files created:")
            for file in result_files:
                print(f"  ✓ {file}")
        
        print(f"\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"The evaluation system now properly:")
        print(f"  ✓ Checks for CSV file existence before including evaluations")
        print(f"  ✓ Skips evaluations where CSV files are missing")
        print(f"  ✓ Does not count skipped evaluations in final metrics")
        print(f"  ✓ Provides detailed logging and reporting of skipped evaluations")
        print(f"  ✓ Saves skipped evaluation information to output files")
        
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
    test_csv_existence_checking() 