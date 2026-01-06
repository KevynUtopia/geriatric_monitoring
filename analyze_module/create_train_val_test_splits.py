#!/usr/bin/env python3
"""
Script to create train/validation/test splits for each person's data.

This script:
1. Groups CSV files by person ID
2. Sorts files chronologically for each person
3. Splits files into train/val/test (7:1:2 ratio) in chronological order
4. Saves the split information to a JSON file for later use
"""

import os
import sys
import glob
import re
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime

def extract_person_id_and_date(filename):
    """
    Extract person ID and date from filename.
    Supports various formats.
    """
    # Remove .csv extension
    basename = filename.replace('.csv', '')
    
    # Pattern 1: ts_recording_YYYY_MM_DD_H_MM_XM_p_N.csv
    pattern1 = r'ts_recording_(\d{4}_\d{2}_\d{2}_\d+_\d+_[ap]m)_p_(\d+)'
    match1 = re.search(pattern1, basename)
    if match1:
        date_info = match1.group(1)
        person_id = f"p_{match1.group(2)}"
        return person_id, date_info
    
    # Pattern 2: p_N.csv (already concatenated files)
    pattern2 = r'^p_(\d+)$'
    match2 = re.search(pattern2, basename)
    if match2:
        person_id = f"p_{match2.group(1)}"
        return person_id, None
        
    # Pattern 3: Other formats with p_N
    pattern3 = r'p_(\d+)'
    match3 = re.search(pattern3, basename)
    if match3:
        person_id = f"p_{match3.group(1)}"
        # Try to extract date if present
        date_match = re.search(r'(\d{4}_\d{2}_\d{2})', basename)
        date_info = date_match.group(1) if date_match else None
        return person_id, date_info
        
    return None, None

def parse_date_for_sorting(date_info):
    """
    Parse date information for chronological sorting.
    Returns a datetime object for comparison.
    """
    if not date_info:
        return datetime.min  # Files without dates go first
    
    try:
        # Pattern: 2019_06_22_9_20_am
        parts = date_info.split('_')
        if len(parts) >= 5:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            
            # Handle am/pm if present
            if len(parts) > 5 and parts[5] in ['am', 'pm']:
                if parts[5] == 'pm' and hour != 12:
                    hour += 12
                elif parts[5] == 'am' and hour == 12:
                    hour = 0
            
            return datetime(year, month, day, hour, minute)
    except:
        pass
    
    return datetime.min

def get_csv_info(csv_path):
    """
    Get basic information about a CSV file.
    """
    try:
        df = pd.read_csv(csv_path, header=0)
        return {
            'path': csv_path,
            'filename': os.path.basename(csv_path),
            'rows': len(df),
            'size_bytes': os.path.getsize(csv_path),
            'columns': list(df.columns)
        }
    except Exception as e:
        return {
            'path': csv_path,
            'filename': os.path.basename(csv_path),
            'rows': 0,
            'size_bytes': os.path.getsize(csv_path),
            'columns': [],
            'error': str(e)
        }

def split_files_chronologically(files, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Split files into train/val/test sets chronologically.
    
    Args:
        files: List of (filepath, filename, date_info) tuples
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
    
    Returns:
        dict: Dictionary with train/val/test file lists
    """
    # Verify ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Sort files chronologically
    def sort_key(file_tuple):
        filepath, filename, date_info = file_tuple
        return parse_date_for_sorting(date_info)
    
    sorted_files = sorted(files, key=sort_key)
    
    n_files = len(sorted_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    n_test = n_files - n_train - n_val  # Remaining files go to test
    
    # Split the files
    train_files = sorted_files[:n_train]
    val_files = sorted_files[n_train:n_train + n_val]
    test_files = sorted_files[n_train + n_val:]
    
    return {
        'train': train_files,
        'val': val_files,
        'test': test_files,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test
    }

def create_train_val_test_splits(data_directory, output_file, min_files=3):
    """
    Create train/val/test splits for all persons and save to file.
    
    Args:
        data_directory: Directory containing CSV files
        output_file: Path to save the split information
        min_files: Minimum number of files required per person
    """
    print("="*80)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("="*80)
    
    print(f"Data directory: {data_directory}")
    print(f"Output file: {output_file}")
    print(f"Minimum files per person: {min_files}")
    print()
    
    # Check if data directory exists
    if not os.path.exists(data_directory):
        print(f"Error: Data directory '{data_directory}' does not exist!")
        return
    
    # Find all CSV files
    csv_pattern = os.path.join(data_directory, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {data_directory}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Group files by person ID
    person_files = defaultdict(list)
    unmatched_files = []
    
    print("Analyzing files...")
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        person_id, date_info = extract_person_id_and_date(filename)
        
        if person_id:
            person_files[person_id].append((csv_file, filename, date_info))
        else:
            unmatched_files.append(filename)
    
    print(f"Processing results:")
    print(f"  Total files: {len(csv_files)}")
    print(f"  Matched files: {len(csv_files) - len(unmatched_files)}")
    print(f"  Unmatched files: {len(unmatched_files)}")
    print(f"  Unique persons found: {len(person_files)}")
    
    # Create splits for each person
    splits_data = {
        'metadata': {
            'data_directory': data_directory,
            'total_files': len(csv_files),
            'matched_files': len(csv_files) - len(unmatched_files),
            'unique_persons': len(person_files),
            'min_files_threshold': min_files,
            'split_ratios': {'train': 0.7, 'val': 0.1, 'test': 0.2}
        },
        'persons': {}
    }
    
    # Sort persons by ID for consistent output
    sorted_persons = sorted(person_files.keys(), key=lambda x: int(x.split('_')[1]))
    
    print(f"\n{'='*80}")
    print("CREATING SPLITS FOR EACH PERSON")
    print(f"{'='*80}")
    
    total_persons_processed = 0
    total_persons_skipped = 0
    
    for person_id in sorted_persons:
        files = person_files[person_id]
        print(f"\nPerson {person_id}: {len(files)} files")
        
        if len(files) < min_files:
            print(f"  ⚠ Skipping {person_id}: Only {len(files)} files (minimum {min_files} required)")
            total_persons_skipped += 1
            continue
        
        # Create splits for this person
        try:
            splits = split_files_chronologically(files)
            
            # Get detailed file information
            split_details = {}
            for split_name in ['train', 'val', 'test']:
                split_files = splits[split_name]
                split_details[split_name] = {
                    'files': [],
                    'count': len(split_files),
                    'total_rows': 0,
                    'total_size_bytes': 0
                }
                
                for filepath, filename, date_info in split_files:
                    file_info = get_csv_info(filepath)
                    file_entry = {
                        'filename': filename,
                        'path': filepath,
                        'date_info': date_info,
                        'rows': file_info['rows'],
                        'size_bytes': file_info['size_bytes']
                    }
                    split_details[split_name]['files'].append(file_entry)
                    split_details[split_name]['total_rows'] += file_info['rows']
                    split_details[split_name]['total_size_bytes'] += file_info['size_bytes']
            
            # Store in main data structure
            splits_data['persons'][person_id] = split_details
            
            # Print split summary
            print(f"  ✓ Train: {splits['n_train']} files ({splits['n_train']/len(files)*100:.1f}%)")
            print(f"  ✓ Val:   {splits['n_val']} files ({splits['n_val']/len(files)*100:.1f}%)")
            print(f"  ✓ Test:  {splits['n_test']} files ({splits['n_test']/len(files)*100:.1f}%)")
            
            total_persons_processed += 1
            
        except Exception as e:
            print(f"  ✗ Error creating splits for {person_id}: {e}")
            total_persons_skipped += 1
    
    # Save splits to file
    try:
        # Create directory only if output_file has a directory component
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(splits_data, f, indent=2)
        print(f"\n✓ Splits saved to: {output_file}")
    except Exception as e:
        print(f"\n✗ Error saving splits: {e}")
        return
    
    # Print final summary
    print(f"\n{'='*80}")
    print("SPLIT CREATION SUMMARY")
    print(f"{'='*80}")
    
    print(f"Total persons found: {len(person_files)}")
    print(f"Persons processed: {total_persons_processed}")
    print(f"Persons skipped: {total_persons_skipped}")
    
    if total_persons_processed > 0:
        print(f"\nSplit distribution across all persons:")
        total_train_files = sum(len(data['train']['files']) for data in splits_data['persons'].values())
        total_val_files = sum(len(data['val']['files']) for data in splits_data['persons'].values())
        total_test_files = sum(len(data['test']['files']) for data in splits_data['persons'].values())
        total_split_files = total_train_files + total_val_files + total_test_files
        
        print(f"  Train: {total_train_files} files ({total_train_files/total_split_files*100:.1f}%)")
        print(f"  Val:   {total_val_files} files ({total_val_files/total_split_files*100:.1f}%)")
        print(f"  Test:  {total_test_files} files ({total_test_files/total_split_files*100:.1f}%)")
        
        print(f"\nExample person splits:")
        for person_id in list(splits_data['persons'].keys())[:3]:  # Show first 3 persons
            data = splits_data['persons'][person_id]
            print(f"  {person_id}: {data['train']['count']}T + {data['val']['count']}V + {data['test']['count']}T")

def load_splits(splits_file):
    """
    Load train/val/test splits from file.
    
    Args:
        splits_file: Path to the splits JSON file
        
    Returns:
        dict: Loaded splits data
    """
    try:
        with open(splits_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading splits from {splits_file}: {e}")
        return None

def main():
    """
    Main function to create the splits.
    """
    print("Train/Val/Test Split Creator")
    print("This script creates chronological file splits for each person")
    print("and saves them for use during training.\n")
    
    # Configuration
    data_directory = "/Users/kevynzhang/Downloads/results_v7_alignment_soft/DATASET"
    output_file = "data_splits.json"
    min_files = 3  # Minimum files needed to create meaningful splits
    
    # Create splits
    create_train_val_test_splits(data_directory, output_file, min_files)
    
    print(f"\nSplits have been saved to: {output_file}")
    print("You can now use these splits for training Factor Analysis models.")

if __name__ == "__main__":
    main() 