import json
import os
import glob
import pandas as pd
from collections import defaultdict
import itertools

def convert_timestamp_to_seconds(timestamp):
    """Convert timestamp from hhmmss format to seconds from start of day"""
    timestamp_str = str(timestamp).zfill(6)  # Ensure 6 digits with leading zeros
    hours = int(timestamp_str[:2])
    minutes = int(timestamp_str[2:4])
    seconds = int(timestamp_str[4:6])
    return hours * 3600 + minutes * 60 + seconds

def convert_seconds_to_timestamp(seconds):
    """Convert seconds from start of day back to hhmmss format"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}{minutes:02d}{secs:02d}"

def forward_fill_interpolation(timestamps, labels, start_time=None, end_time=None):
    """
    Forward fill interpolation between timestamps
    Returns arrays for continuous processing
    """
    if not timestamps or not labels:
        return [], []
    
    # Convert timestamps to seconds
    time_seconds = [convert_timestamp_to_seconds(ts) for ts in timestamps]
    
    # Determine the range
    if start_time is None:
        start_time = min(time_seconds)
    if end_time is None:
        end_time = max(time_seconds)
    
    # Create continuous time array
    continuous_time = list(range(start_time, end_time + 1))
    continuous_labels = []
    
    current_label = 0  # Default starting label
    ts_index = 0
    
    for t in continuous_time:
        # Check if we need to update the label
        while ts_index < len(time_seconds) and time_seconds[ts_index] <= t:
            current_label = labels[ts_index]
            ts_index += 1
        continuous_labels.append(current_label)
    
    return continuous_time, continuous_labels

def aggregate_person_results(person_key, evaluator_data, global_start, global_end):
    """Aggregate results for a single person from all evaluators"""
    # Dictionary to store aggregated data: {second: {'results': sum, 'count': count}}
    aggregated = defaultdict(lambda: {'results': 0, 'count': 0})
    
    # Track if any data was found for this person
    data_found = False
    
    # Process data from each evaluator for this person
    for evaluator_name, evaluator_info in evaluator_data.items():
        results = evaluator_info['results']
        
        # Check if this evaluator has data for this person
        if person_key in results:
            p_data = results[person_key]
            
            if 'timestamps' in p_data and 'labels' in p_data:
                timestamps = p_data['timestamps']
                labels = p_data['labels']
                
                if timestamps and labels:
                    # Forward fill interpolation
                    cont_time, cont_labels = forward_fill_interpolation(
                        timestamps, labels, global_start, global_end
                    )
                    
                    # Add each timestamp's results to the aggregation
                    for time_sec, label in zip(cont_time, cont_labels):
                        aggregated[time_sec]['results'] += label
                        aggregated[time_sec]['count'] += 1
                    
                    data_found = True
    
    return aggregated, data_found

def save_person_csv(person_key, aggregated_data, output_dir, filename):
    """Save aggregated results for a single person to CSV file"""
    # Prepare data for CSV
    csv_data = []
    
    # Sort by timestamp (second)
    sorted_times = sorted(aggregated_data.keys())
    
    for time_sec in sorted_times:
        data = aggregated_data[time_sec]
        csv_data.append({
            'timestamp': convert_seconds_to_timestamp(time_sec),
            'second': time_sec,
            'results': data['results'],
            'count': data['count']
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    
    # Create output path for this person
    person_output_path = os.path.join(output_dir, f'{person_key}.csv')
    df.to_csv(person_output_path, index=False)
    
    print(f"Saved {person_key}: {len(csv_data)} data points")
    print(f"  Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"  Results: min={df['results'].min()}, max={df['results'].max()}, mean={df['results'].mean():.2f}")
    print(f"  Count: min={df['count'].min()}, max={df['count'].max()}, mean={df['count'].mean():.2f}")
    
    return person_output_path

def process_json_file(json_file_path, base_output_dir, included_annotators=None):
    """Process a single JSON file and create unified CSV files for each person, for a given set of annotators"""
    
    try:
        # Load JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract filename for title and folder creation
        filename = os.path.basename(json_file_path).replace('.json', '')
        
        # Get all evaluator names (primary keys) and organize their data
        evaluator_data = {}
        for evaluator_name in data.keys():
            if included_annotators is not None and evaluator_name not in included_annotators:
                continue
            if 'results' in data[evaluator_name]:
                evaluator_data[evaluator_name] = {
                    'results': data[evaluator_name]['results'],
                    'cam': data[evaluator_name].get('cam', 'unknown')
                }
        
        # Collect all unique person keys (p_1, p_2, etc.) across all evaluators
        all_person_keys = set()
        for evaluator_info in evaluator_data.values():
            all_person_keys.update(evaluator_info['results'].keys())
        
        # Sort person keys naturally
        person_keys = sorted(all_person_keys, key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else float('inf'))
        
        if not person_keys:
            print(f"No person data found in {filename}")
            return
        
        # Find global time range for consistent processing
        all_timestamps = []
        for evaluator_info in evaluator_data.values():
            for p_key, p_data in evaluator_info['results'].items():
                if 'timestamps' in p_data:
                    all_timestamps.extend(p_data['timestamps'])
        
        if all_timestamps:
            global_start = min([convert_timestamp_to_seconds(ts) for ts in all_timestamps])
            global_end = max([convert_timestamp_to_seconds(ts) for ts in all_timestamps])
        else:
            print(f"No timestamps found in {filename}")
            return
        
        # Create output directory for this JSON file (same structure as visualization)
        json_output_dir = os.path.join(base_output_dir, filename)
        os.makedirs(json_output_dir, exist_ok=True)
        
        print(f"Processing {filename}: {len(person_keys)} persons, {len(evaluator_data)} evaluators")
        
        # Process each person separately and create individual CSV files
        csv_files_created = 0
        for person_key in person_keys:
            # Aggregate results for this person from all evaluators
            aggregated_data, data_found = aggregate_person_results(
                person_key, evaluator_data, global_start, global_end
            )
            
            if data_found and aggregated_data:
                # Save CSV file for this person
                csv_path = save_person_csv(person_key, aggregated_data, json_output_dir, filename)
                csv_files_created += 1
            else:
                print(f"No data found for {person_key}")
        
        print(f"Created {csv_files_created} CSV files in {json_output_dir}")
        
    except Exception as e:
        print(f"Error processing {json_file_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def process_txt_file(txt_file_path, base_output_dir):
    """Process a single TXT file and create unified CSV file"""
    
    try:
        # Extract filename for identification
        filename = os.path.basename(txt_file_path).replace('.txt', '')
        
        # Read the txt file content
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse the content (assuming similar structure to JSON but in text format)
        timestamps = []
        labels = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Assuming format: timestamp,label or timestamp label
                if ',' in line:
                    parts = line.split(',')
                elif ' ' in line:
                    parts = line.split()
                else:
                    continue
                
                if len(parts) >= 2:
                    try:
                        timestamp = int(parts[0])
                        label = int(parts[1])
                        timestamps.append(timestamp)
                        labels.append(label)
                    except ValueError:
                        continue
        
        if timestamps and labels:
            # Find time range
            time_seconds = [convert_timestamp_to_seconds(ts) for ts in timestamps]
            global_start = min(time_seconds)
            global_end = max(time_seconds)
            
            # Forward fill interpolation
            cont_time, cont_labels = forward_fill_interpolation(
                timestamps, labels, global_start, global_end
            )
            
            # Create aggregated data structure
            aggregated = defaultdict(lambda: {'results': 0, 'count': 0})
            for time_sec, label in zip(cont_time, cont_labels):
                aggregated[time_sec]['results'] += label
                aggregated[time_sec]['count'] += 1
            
            # Create output directory for this TXT file
            txt_output_dir = os.path.join(base_output_dir, filename)
            os.makedirs(txt_output_dir, exist_ok=True)
            
            # Save CSV file for TXT data
            csv_path = save_person_csv("txt_data", aggregated, txt_output_dir, filename)
            print(f"Created CSV file for TXT data in {txt_output_dir}")
            
        else:
            print(f"No valid data found in {filename}")
        
    except Exception as e:
        print(f"Error processing {txt_file_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def process_all_files(base_path, base_output_dir, included_annotators=None):
    """Process all JSON and TXT files in the given directory, for a given set of annotators"""
    
    # Find all JSON and TXT files
    json_pattern = os.path.join(base_path, "*.json")
    txt_pattern = os.path.join(base_path, "*.txt")
    
    json_files = glob.glob(json_pattern)
    txt_files = glob.glob(txt_pattern)
    
    all_files = json_files + txt_files
    
    if not all_files:
        print(f"No JSON or TXT files found in {base_path}")
        return
    
    print(f"Found {len(json_files)} JSON files and {len(txt_files)} TXT files")
    
    # Process JSON files
    for json_file in sorted(json_files):
        print(f"\nProcessing JSON: {os.path.basename(json_file)}")
        process_json_file(json_file, base_output_dir, included_annotators)
    
    # Process TXT files
    for txt_file in sorted(txt_files):
        print(f"\nProcessing TXT: {os.path.basename(txt_file)}")
        process_txt_file(txt_file, base_output_dir)

def main():
    # Configuration - using same structure as visualization script
    base_path = "path_to_your_analysis_root/SNH/human_evaluation"  # Same as visualization script
    annotators = ["e3", "e5", "e4", "e2", "e1"]  # Anonymized: e1=cyy, e2=zzr, e3=hsy, e4=zww, e5=roy
    print("Starting Human Evaluation Results Unification for all annotator combinations")
    print(f"Looking for JSON files in: {base_path}")
    print("Creating separate CSV files for each person in each recording session...")
    combo_root_dir = "path_to_your_analysis_root/SNH/human_evaluation_combo"
    os.makedirs(combo_root_dir, exist_ok=True)
    # Generate all non-empty combinations of annotators
    for r in range(1, len(annotators)+1):
        for combo in itertools.combinations(annotators, r):
            combo_sorted = sorted(combo)
            combo_name = "_".join(combo_sorted)
            base_output_dir = os.path.join(combo_root_dir, f"human_evaluation_unified_{combo_name}")
            print(f"\n=== Processing combination: {combo_name} ===")
            os.makedirs(base_output_dir, exist_ok=True)
            process_all_files(base_path, base_output_dir, included_annotators=combo_sorted)
    print("\nUnification for all combinations complete!")
    print(f"CSV files saved in subfolders of: {combo_root_dir}")

if __name__ == "__main__":
    main()
