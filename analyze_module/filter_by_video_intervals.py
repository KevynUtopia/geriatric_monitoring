import json
import os
import glob
import pandas as pd
import re
from collections import defaultdict

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

def parse_video_filename(filename):
    """Parse video filename to extract camera and time interval"""
    # Pattern: cam_10-00120-00240.mp4
    pattern = r'cam_(\d+)-(\d+)-(\d+)\.mp4'
    match = re.match(pattern, filename)
    if match:
        cam_num = int(match.group(1))
        start_time = int(match.group(2))
        end_time = int(match.group(3))
        return cam_num, start_time, end_time
    return None, None, None

def get_video_intervals(split_videos_base_path):
    """Get all available video intervals for each recording session and camera"""
    video_intervals = {}
    
    # Get all recording session directories
    recording_dirs = glob.glob(os.path.join(split_videos_base_path, "recording_*"))
    
    for recording_dir in recording_dirs:
        if 'zip' in recording_dir:
            continue
        recording_name = os.path.basename(recording_dir)
        video_intervals[recording_name] = defaultdict(list)
        
        # Get all camera directories in this recording
        cam_dirs = glob.glob(os.path.join(recording_dir, "cam_*"))
        
        for cam_dir in cam_dirs:
            cam_name = os.path.basename(cam_dir)
            
            # Get all video files in this camera directory
            video_files = glob.glob(os.path.join(cam_dir, "*.mp4"))
            
            for video_file in video_files:
                filename = os.path.basename(video_file)
                cam_num, start_time, end_time = parse_video_filename(filename)
                
                if cam_num is not None:
                    video_intervals[recording_name][cam_name].append((start_time, end_time))
            
            # Sort intervals by start time
            video_intervals[recording_name][cam_name].sort()
    
    return video_intervals

def load_camera_start_times(evaluation_head_path):
    """Load camera start times from evaluation_head.json"""
    with open(evaluation_head_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_relative_to_absolute_time(relative_seconds, cam_start_time):
    """Convert relative seconds (from video start) to absolute seconds (from day start)"""
    cam_start_seconds = convert_timestamp_to_seconds(cam_start_time)
    return cam_start_seconds + relative_seconds

def is_time_in_video_intervals(absolute_second, video_intervals, cam_start_time):
    """Check if a given absolute time falls within any video interval"""
    cam_start_seconds = convert_timestamp_to_seconds(cam_start_time)
    
    # Convert absolute time back to relative time (from camera start)
    relative_second = absolute_second - cam_start_seconds
    
    # Check if this relative time falls within any video interval
    for start_time, end_time in video_intervals:
        if start_time <= relative_second <= end_time:
            return True
    
    return False

def filter_csv_by_video_intervals(csv_path, video_intervals, cam_start_times, recording_name):
    """Filter a CSV file to only include times that fall within video intervals"""
    df = pd.read_csv(csv_path)
    
    # Get the person key from filename (p_1.csv -> p_1)
    person_key = os.path.basename(csv_path).replace('.csv', '')
    
    # We need to determine which camera this person corresponds to
    # For now, we'll check all cameras and include times that exist in any camera
    # This is a simplification - in reality, you might need person-to-camera mapping
    
    filtered_rows = []
    
    for _, row in df.iterrows():
        absolute_second = int(row['second'])
        timestamp = row['timestamp']
        
        # Check if this time falls within any camera's video intervals
        time_found_in_any_camera = False
        
        for cam_name in video_intervals.get(recording_name, {}):
            if cam_name in cam_start_times.get(recording_name, {}):
                cam_start_time = cam_start_times[recording_name][cam_name]
                cam_intervals = video_intervals[recording_name][cam_name]
                
                if is_time_in_video_intervals(absolute_second, cam_intervals, cam_start_time):
                    time_found_in_any_camera = True
                    break
        
        if time_found_in_any_camera:
            filtered_rows.append(row)
    
    return pd.DataFrame(filtered_rows)

def process_recording_session(recording_name, unified_results_path, output_path, video_intervals, cam_start_times):
    """Process all CSV files for a single recording session"""
    
    # Create output directory
    session_output_dir = os.path.join(output_path, recording_name)
    os.makedirs(session_output_dir, exist_ok=True)
    
    # Get all CSV files in this recording session
    session_input_dir = os.path.join(unified_results_path, recording_name)
    if not os.path.exists(session_input_dir):
        print(f"No unified results found for {recording_name}")
        return
    
    csv_files = glob.glob(os.path.join(session_input_dir, "*.csv"))
    
    total_original_rows = 0
    total_filtered_rows = 0
    
    for csv_file in csv_files:
        person_key = os.path.basename(csv_file).replace('.csv', '')
        
        # Read original CSV
        original_df = pd.read_csv(csv_file)
        total_original_rows += len(original_df)
        
        # Filter based on video intervals
        filtered_df = filter_csv_by_video_intervals(csv_file, video_intervals, cam_start_times, recording_name)
        total_filtered_rows += len(filtered_df)
        
        # Save filtered CSV
        output_csv_path = os.path.join(session_output_dir, f"{person_key}.csv")
        filtered_df.to_csv(output_csv_path, index=False)
        
        print(f"  {person_key}: {len(original_df)} -> {len(filtered_df)} rows ({len(filtered_df)/len(original_df)*100:.1f}% retained)")
    
    print(f"  Total: {total_original_rows} -> {total_filtered_rows} rows ({total_filtered_rows/total_original_rows*100:.1f}% retained)")

def filter_all_combo_folders(combo_root_dir, combo_filtered_root_dir, video_intervals, cam_start_times):
    """Filter all combo folders in combo_root_dir and output to combo_filtered_root_dir"""
    if not os.path.exists(combo_root_dir):
        print(f"Combo root directory not found: {combo_root_dir}")
        return
    combo_folders = [d for d in os.listdir(combo_root_dir) if os.path.isdir(os.path.join(combo_root_dir, d))]
    print(f"\nFiltering all combo folders in {combo_root_dir}...")
    for combo_folder in combo_folders:
        unified_results_path = os.path.join(combo_root_dir, combo_folder)
        output_path = os.path.join(combo_filtered_root_dir, combo_folder)
        os.makedirs(output_path, exist_ok=True)
        print(f"\nFiltering combo: {combo_folder}")
        # For each recording session in this combo folder
        for recording_name in video_intervals.keys():
            print(f"  Processing {recording_name}...")
            process_recording_session(recording_name, unified_results_path, output_path, video_intervals, cam_start_times)
    print(f"\nFiltering of all combos complete! Results saved to: {combo_filtered_root_dir}")


def main():
    # Configuration
    split_videos_base_path = "path_to_your_analysis_root/SNH/split_videos"
    unified_results_path = "path_to_your_analysis_root/SNH/human_evaluation_unified"
    evaluation_head_path = "path_to_your_analysis_root/SNH/human_evaluation_unified/evaluation_head.json"
    output_path = "path_to_your_analysis_root/SNH/human_evaluation_filtered"
    combo_root_dir = "path_to_your_analysis_root/SNH/human_evaluation_combo"
    combo_filtered_root_dir = "path_to_your_analysis_root/SNH/human_evaluation_combo_filtered"
    print("Starting filtering process...")
    print(f"Split videos path: {split_videos_base_path}")
    print(f"Unified results path: {unified_results_path}")
    print(f"Output path: {output_path}")
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    # Load camera start times
    print("\nLoading camera start times...")
    cam_start_times = load_camera_start_times(evaluation_head_path)
    # Get video intervals for all recording sessions
    print("\nAnalyzing video intervals...")
    video_intervals = get_video_intervals(split_videos_base_path)
    # Print summary of available video intervals
    print("\nAvailable video intervals:")
    for recording_name, cameras in video_intervals.items():
        print(f"  {recording_name}:")
        for cam_name, intervals in cameras.items():
            total_duration = sum(end - start for start, end in intervals)
            print(f"    {cam_name}: {len(intervals)} intervals, {total_duration} seconds total")
    # Process each recording session for the main unified results
    print("\nFiltering evaluation results...")
    for recording_name in video_intervals.keys():
        print(f"\nProcessing {recording_name}...")
        process_recording_session(recording_name, unified_results_path, output_path, video_intervals, cam_start_times)
    print(f"\nFiltering complete! Results saved to: {output_path}")
    # Now filter all combo folders
    filter_all_combo_folders(combo_root_dir, combo_filtered_root_dir, video_intervals, cam_start_times)

if __name__ == "__main__":
    main() 