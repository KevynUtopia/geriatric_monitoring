import json
import re
import os
from typing import Dict, Tuple, List, Any
import argparse

def read_results_e3(file_path: str) -> Dict[str, Tuple[List[int], List[int]]]:  # Anonymized: e3=hsy
    """
    Read the results JSON file and convert it into a dictionary with timestamps and states.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    result_dict = {}
    
    for person_id, timestamps in data.items():
        # Sort timestamps to maintain order
        sorted_timestamps = sorted(timestamps.items(), key=lambda x: int(x[0]))
        
        # Extract timestamps and convert to 6-digit integers
        timestamp_list = [int(ts) for ts, _ in sorted_timestamps]
        
        # Convert labels to 1 (anomaly) and 0 (normal)
        label_list = [1 if label == "anomaly" else 0 for _, label in sorted_timestamps]
        
        result_dict[person_id] = (timestamp_list, label_list)
    
    return result_dict

def read_results_e5(file_path: str) -> Dict[str, Tuple[List[int], List[int]]]:  # Anonymized: e5=roy
    """
    Read the results JSON file and convert it into a dictionary with timestamps and states.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    result_dict = {}
    
    # Extract data from the 'e5' key (anonymized: e5=roy)
    roy_data = data.get('e5', {})
    
    for person_id, timestamps in roy_data.items():
        # Sort timestamps to maintain order
        sorted_timestamps = sorted(timestamps.items(), key=lambda x: int(x[0]))
        
        # Extract timestamps and convert to 6-digit integers
        timestamp_list = [int(ts) for ts, _ in sorted_timestamps]
        
        # Convert labels to 1 (ANM) and 0 (N)
        label_list = [1 if label == "ANM" else 0 for _, label in sorted_timestamps]
        
        result_dict[person_id] = (timestamp_list, label_list)
    
    return result_dict

def read_results_e4(file_path: str) -> Dict[str, Tuple[List[int], List[int]]]:  # Anonymized: e4=zww
    """
    Read the results JSON file and convert it into a dictionary with timestamps and anomaly labels.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    result_dict = {}
    
    for person_id, timestamps in data.items():
        # Check if timestamps is a dictionary
        if isinstance(timestamps, dict):
            # Sort timestamps to maintain order
            sorted_timestamps = sorted(timestamps.items(), key=lambda x: int(x[0]))
            
            # Extract timestamps and convert to 6-digit integers
            timestamp_list = [int(ts.zfill(6)) for ts, _ in sorted_timestamps]
            
            # Convert labels to 1 (anomaly) and 0 (normal)
            label_list = [1 if label == "anomaly" else 0 for _, label in sorted_timestamps]
            
            result_dict[person_id] = (timestamp_list, label_list)
        else:
            print(f"Warning: Unexpected data type for {person_id}: {type(timestamps)}")
            continue
    
    return result_dict

def read_results_e2(file_path: str) -> Dict[str, Tuple[List[int], List[int]]]:  # Anonymized: e2=zzr
    """
    Read the results TXT file and convert it into a dictionary with timestamps and states.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    result_dict = {}
    
    # separate the content by \n, and remove the first and last element
    content = content.split('\n')[1:-1]
    for line in content:
        # remove spaces and :
        line = line.replace(' ', '').replace(':', '')
        
        # Extract person ID and their data
        match = re.match(r'([^{]+)\{([^}]+)\}', line)
        if match:
            person_id = match.group(1)
            data_str = match.group(2)
            
            # Parse the timestamp-state pairs
            pairs = re.findall(r'\(([^,]+),([^)]+)\)', data_str)
            
            # Create lists for timestamps and states
            timestamps = []
            states = []
            
            # Sort pairs by timestamp to maintain order
            sorted_pairs = sorted(pairs, key=lambda x: int(x[0].zfill(6)))
            
            for timestamp, state in sorted_pairs:
                # Convert timestamp to 6-digit integer
                timestamp_int = int(timestamp.zfill(6))
                # Convert state to binary (0 for N, 1 for ANM)
                state_int = 1 if state == 'ANM' else 0
                timestamps.append(timestamp_int)
                states.append(state_int)
            
            result_dict[person_id] = (timestamps, states)
    
    return result_dict

def read_results_e1(file_path: str) -> Dict[str, Tuple[List[int], List[int]]]:  # Anonymized: e1=cyy
    """
    Read the results JSON file and convert it into a dictionary with timestamps and anomaly labels.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    result_dict = {}
    
    for person_id, timestamps in data.items():
        # Sort timestamps to maintain order
        sorted_timestamps = sorted(timestamps.items(), key=lambda x: int(x[0]))
        
        # Extract timestamps and convert to 6-digit integer
        timestamp_list = [int(ts.zfill(6)) for ts, _ in sorted_timestamps]
        
        # Convert labels to 1 (anomaly) and 0 (normal)
        label_list = [1 if label == "anomaly" else 0 for _, label in sorted_timestamps]
        
        result_dict[person_id] = (timestamp_list, label_list)
    
    return result_dict

def extract_camera_info(filename: str) -> Tuple[str, str]:
    """
    Extract camera number and reader type from filename.
    
    Returns:
        Tuple[str, str]: (camera_number, reader_type)
    """
    # Extract camera number (cam_XX or camXX)
    cam_match = re.search(r'cam_?(\d+)', filename.lower())
    camera = cam_match.group(1) if cam_match else "unknown"
    
    # Extract reader type (anonymized: e1=cyy, e2=zzr, e3=hsy, e4=zww, e5=roy)
    reader_type = None
    for reader in ['e3', 'e5', 'e4', 'e2', 'e1']:  # Anonymized annotators
        if reader in filename.lower():
            reader_type = reader
            break
    
    return camera, reader_type

def convert_results_to_serializable(results: Dict[str, Tuple[List[int], List[int]]]) -> Dict[str, Dict[str, List[int]]]:
    """
    Convert results format to JSON-serializable format.
    """
    serializable_results = {}
    for person_id, (timestamps, labels) in results.items():
        serializable_results[person_id] = {
            "timestamps": timestamps,
            "labels": labels
        }
    return serializable_results

def process_folder(folder_path: str) -> Dict[str, Any]:
    """
    Process all files in a folder and return unified results.
    """
    unified_data = {}
    
    # Get all files in the folder
    try:
        files_in_folder = os.listdir(folder_path)
    except Exception as e:
        print(f"Error accessing folder {folder_path}: {e}")
        return {}
    
    # Process each file
    for filename in files_in_folder:
        file_path = os.path.join(folder_path, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        camera, reader_type = extract_camera_info(filename)
        
        if reader_type is None:
            print(f"Skipping file {filename}: Unknown reader type")
            continue
        
        print(f"Processing {filename} with {reader_type} reader (camera {camera})")
        
        try:
            # Choose appropriate reader function (anonymized: e1=cyy, e2=zzr, e3=hsy, e4=zww, e5=roy)
            if reader_type == 'e3':
                results = read_results_e3(file_path)
            elif reader_type == 'e5':
                results = read_results_e5(file_path)
            elif reader_type == 'e4':
                results = read_results_e4(file_path)
            elif reader_type == 'e2':
                results = read_results_e2(file_path)
            elif reader_type == 'e1':
                results = read_results_e1(file_path)
            else:
                print(f"Unknown reader type: {reader_type}")
                continue
            
            # Convert to serializable format
            serializable_results = convert_results_to_serializable(results)
            
            # Add to unified data
            unified_data[reader_type] = {
                "cam": camera,
                "results": serializable_results
            }
            
            print(f"Successfully processed {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return unified_data

def extract_recording_name(folder_path: str) -> str:
    """
    Extract recording session name from folder path.
    """
    folder_name = os.path.basename(folder_path.rstrip('/'))
    
    # If folder name already looks like a recording name, use it
    if 'recording_' in folder_name:
        return folder_name
    
    # Otherwise, create a generic name
    return f"recording_{folder_name}"

def main():
    parser = argparse.ArgumentParser(description='Unified result reader for all file types')
    parser.add_argument('--folder_path', default="path_to_your_analysis_root/SNH/recording_labels", help='Path to root folder containing recording folders')
    parser.add_argument('--output_dir', default="path_to_your_analysis_root/SNH/human_evaluation", help='Output directory for JSON files')
    
    args = parser.parse_args()
    
    root_folder_path = args.folder_path
    output_dir = args.output_dir
    
    if not os.path.exists(root_folder_path):
        print(f"Error: Root folder {root_folder_path} does not exist")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print(f"Processing root folder: {root_folder_path}")
    print(f"Output directory: {output_dir}")
    
    # Get all folders under root_folder_path
    try:
        all_items = os.listdir(root_folder_path)
        folders = [item for item in all_items if os.path.isdir(os.path.join(root_folder_path, item))]
    except Exception as e:
        print(f"Error accessing root folder {root_folder_path}: {e}")
        return
    
    if not folders:
        print("No subfolders found in the root directory")
        return
    
    print(f"Found {len(folders)} folders to process")
    
    processed_count = 0
    for folder_name in folders:
        folder_path = os.path.join(root_folder_path, folder_name)
        print(f"\n{'='*60}")
        print(f"Processing folder: {folder_name}")
        print(f"Full path: {folder_path}")
        
        # Process the folder
        unified_data = process_folder(folder_path)
        
        if not unified_data:
            print(f"No valid data found in folder: {folder_name}")
            continue
        
        # Generate output filename based on folder name
        recording_name = extract_recording_name(folder_path)
        output_filename = os.path.join(output_dir, f"{recording_name}.json")
        
        # Save unified results
        try:
            with open(output_filename, 'w') as f:
                json.dump(unified_data, f, indent=2)
            print(f"Unified results saved to: {output_filename}")
            
            # Print summary for this folder
            print(f"Summary for {folder_name}:")
            for reader_type, data in unified_data.items():
                cam_number = data['cam']
                person_count = len(data['results'])
                print(f"  {reader_type} (camera {cam_number}): {person_count} persons")
            
            processed_count += 1
                
        except Exception as e:
            print(f"Error saving results for {folder_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed {processed_count} out of {len(folders)} folders")
    print(f"Output files saved in: {output_dir}")

if __name__ == "__main__":
    main() 