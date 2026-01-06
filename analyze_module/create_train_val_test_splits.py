#!/usr/bin/env python3
"""Create train/validation/test splits for each person's data."""

import os
import glob
import re
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime

def extract_person_id_and_date(filename):
    """Extract person ID and date from filename."""
    basename = filename.replace('.csv', '')
    
    pattern1 = r'ts_recording_(\d{4}_\d{2}_\d{2}_\d+_\d+_[ap]m)_p_(\d+)'
    match1 = re.search(pattern1, basename)
    if match1:
        date_info = match1.group(1)
        person_id = f"p_{match1.group(2)}"
        return person_id, date_info
    
    pattern2 = r'^p_(\d+)$'
    match2 = re.search(pattern2, basename)
    if match2:
        person_id = f"p_{match2.group(1)}"
        return person_id, None
        
    pattern3 = r'p_(\d+)'
    match3 = re.search(pattern3, basename)
    if match3:
        person_id = f"p_{match3.group(1)}"
        date_match = re.search(r'(\d{4}_\d{2}_\d{2})', basename)
        date_info = date_match.group(1) if date_match else None
        return person_id, date_info
        
    return None, None

def parse_date_for_sorting(date_info):
    """Parse date information for chronological sorting."""
    if not date_info:
        return datetime.min
    
    try:
        parts = date_info.split('_')
        if len(parts) >= 5:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            
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
    """Get basic information about a CSV file."""
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
    """Split files into train/val/test sets chronologically."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    def sort_key(file_tuple):
        filepath, filename, date_info = file_tuple
        return parse_date_for_sorting(date_info)
    
    sorted_files = sorted(files, key=sort_key)
    n_files = len(sorted_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    n_test = n_files - n_train - n_val
    
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
    """Create train/val/test splits for all persons and save to file."""
    if not os.path.exists(data_directory):
        print(f"Error: Data directory '{data_directory}' does not exist!")
        return
    
    csv_pattern = os.path.join(data_directory, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {data_directory}")
        return
    
    person_files = defaultdict(list)
    unmatched_files = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        person_id, date_info = extract_person_id_and_date(filename)
        
        if person_id:
            person_files[person_id].append((csv_file, filename, date_info))
        else:
            unmatched_files.append(filename)
    
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
    
    sorted_persons = sorted(person_files.keys(), key=lambda x: int(x.split('_')[1]))
    total_persons_processed = 0
    total_persons_skipped = 0
    
    for person_id in sorted_persons:
        files = person_files[person_id]
        
        if len(files) < min_files:
            total_persons_skipped += 1
            continue
        
        try:
            splits = split_files_chronologically(files)
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
            
            splits_data['persons'][person_id] = split_details
            total_persons_processed += 1
            
        except Exception as e:
            print(f"Error creating splits for {person_id}: {e}")
            total_persons_skipped += 1
    
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(splits_data, f, indent=2)
        print(f"Splits saved to: {output_file}")
    except Exception as e:
        print(f"Error saving splits: {e}")
        return
    
    print(f"Processed {total_persons_processed} persons, skipped {total_persons_skipped}")

def load_splits(splits_file):
    """Load train/val/test splits from file."""
    try:
        with open(splits_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading splits from {splits_file}: {e}")
        return None

def main():
    data_directory = "/Users/kevynzhang/Downloads/results_alignment_soft/DATASET"
    output_file = "data_splits.json"
    min_files = 3
    create_train_val_test_splits(data_directory, output_file, min_files)

if __name__ == "__main__":
    main() 