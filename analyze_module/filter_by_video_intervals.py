import json
import os
import glob
import pandas as pd
import re
from collections import defaultdict

def convert_timestamp_to_seconds(timestamp):
    """Convert timestamp from hhmmss format to seconds from start of day"""
    timestamp_str = str(timestamp).zfill(6)
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
    recording_dirs = glob.glob(os.path.join(split_videos_base_path, "recording_*"))
    
    for recording_dir in recording_dirs:
        if 'zip' in recording_dir:
            continue
        recording_name = os.path.basename(recording_dir)
        video_intervals[recording_name] = defaultdict(list)
        
        cam_dirs = glob.glob(os.path.join(recording_dir, "cam_*"))
        for cam_dir in cam_dirs:
            cam_name = os.path.basename(cam_dir)
            video_files = glob.glob(os.path.join(cam_dir, "*.mp4"))
            
            for video_file in video_files:
                filename = os.path.basename(video_file)
                cam_num, start_time, end_time = parse_video_filename(filename)
                
                if cam_num is not None:
                    video_intervals[recording_name][cam_name].append((start_time, end_time))
            
            video_intervals[recording_name][cam_name].sort()
    
    return video_intervals

def load_camera_start_times(evaluation_head_path):
    """Load camera start times from evaluation_head.json"""
    with open(evaluation_head_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_time_in_video_intervals(absolute_second, video_intervals, cam_start_time):
    """Check if a given absolute time falls within any video interval"""
    cam_start_seconds = convert_timestamp_to_seconds(cam_start_time)
    relative_second = absolute_second - cam_start_seconds
    
    for start_time, end_time in video_intervals:
        if start_time <= relative_second <= end_time:
            return True
    return False

def filter_csv_by_video_intervals(csv_path, video_intervals, cam_start_times, recording_name):
    """Filter a CSV file to only include times that fall within video intervals"""
    df = pd.read_csv(csv_path)
    filtered_rows = []
    
    for _, row in df.iterrows():
        absolute_second = int(row['second'])
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
    session_output_dir = os.path.join(output_path, recording_name)
    os.makedirs(session_output_dir, exist_ok=True)
    
    session_input_dir = os.path.join(unified_results_path, recording_name)
    if not os.path.exists(session_input_dir):
        return
    
    csv_files = glob.glob(os.path.join(session_input_dir, "*.csv"))
    
    for csv_file in csv_files:
        person_key = os.path.basename(csv_file).replace('.csv', '')
        filtered_df = filter_csv_by_video_intervals(csv_file, video_intervals, cam_start_times, recording_name)
        output_csv_path = os.path.join(session_output_dir, f"{person_key}.csv")
        filtered_df.to_csv(output_csv_path, index=False)

def filter_all_combo_folders(combo_root_dir, combo_filtered_root_dir, video_intervals, cam_start_times):
    """Filter all combo folders in combo_root_dir and output to combo_filtered_root_dir"""
    if not os.path.exists(combo_root_dir):
        return
    combo_folders = [d for d in os.listdir(combo_root_dir) if os.path.isdir(os.path.join(combo_root_dir, d))]
    
    for combo_folder in combo_folders:
        unified_results_path = os.path.join(combo_root_dir, combo_folder)
        output_path = os.path.join(combo_filtered_root_dir, combo_folder)
        os.makedirs(output_path, exist_ok=True)
        
        for recording_name in video_intervals.keys():
            process_recording_session(recording_name, unified_results_path, output_path, video_intervals, cam_start_times)


def main():
    split_videos_base_path = "path_to_your_analysis_root/SNH/split_videos"
    unified_results_path = "path_to_your_analysis_root/SNH/human_evaluation_unified"
    evaluation_head_path = "path_to_your_analysis_root/SNH/human_evaluation_unified/evaluation_head.json"
    output_path = "path_to_your_analysis_root/SNH/human_evaluation_filtered"
    combo_root_dir = "path_to_your_analysis_root/SNH/human_evaluation_combo"
    combo_filtered_root_dir = "path_to_your_analysis_root/SNH/human_evaluation_combo_filtered"
    
    os.makedirs(output_path, exist_ok=True)
    cam_start_times = load_camera_start_times(evaluation_head_path)
    video_intervals = get_video_intervals(split_videos_base_path)
    
    for recording_name in video_intervals.keys():
        process_recording_session(recording_name, unified_results_path, output_path, video_intervals, cam_start_times)
    
    filter_all_combo_folders(combo_root_dir, combo_filtered_root_dir, video_intervals, cam_start_times)

if __name__ == "__main__":
    main() 