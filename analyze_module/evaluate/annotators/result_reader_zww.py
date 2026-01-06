import json
from typing import Dict, Tuple, List
import os

def read_results_e4(file_path: str) -> Dict[str, Tuple[List[str], List[int]]]:  # Anonymized: e4=zww
    """
    Read the results JSON file and convert it into a dictionary with timestamps and anomaly labels.
    
    Args:
        file_path (str): Path to the JSON file containing the results
        
    Returns:
        Dict[str, Tuple[List[str], List[int]]]: Dictionary where:
            - Keys are the person IDs (e.g., 'p_4', 'p_7', etc.)
            - Values are tuples containing:
                - List of timestamps (6-digit strings)
                - List of anomaly labels (1 for anomaly, 0 for normal)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    result_dict = {}
    
    for person_id, timestamps in data.items():
        # Debug: print the type and content of timestamps
        # print(f"Debug - Person {person_id}: type={type(timestamps)}, content={timestamps}")
        
        # Check if timestamps is a dictionary
        if isinstance(timestamps, dict):
            # Sort timestamps to maintain order
            sorted_timestamps = sorted(timestamps.items(), key=lambda x: int(x[0]))
            
            # Extract timestamps and convert to 6-digit strings
            timestamp_list = [int(ts.zfill(6)) for ts, _ in sorted_timestamps]
            
            # Convert labels to 1 (anomaly) and 0 (normal)
            label_list = [1 if label == "anomaly" else 0 for _, label in sorted_timestamps]
            
            result_dict[person_id] = (timestamp_list, label_list)
        else:
            print(f"Warning: Unexpected data type for {person_id}: {type(timestamps)}")
            # Skip this person if data format is unexpected
            continue
    
    return result_dict

if __name__ == "__main__":
    # Example usage
    folder_path = "path_to_your_analysis_root/SNH/recording_labels"
    
    # Get all folders under folder_path
    all_items = os.listdir(folder_path)
    folders = [item for item in all_items if os.path.isdir(os.path.join(folder_path, item))]
    
    for folder in folders:
        folder_full_path = os.path.join(folder_path, folder)
        print(f"\nChecking folder: {folder}")
        
        # Get all files in this folder
        try:
            files_in_folder = os.listdir(folder_full_path)
            # Find files containing "e4" or "zww" in their names (e4=zww anonymized)
            zww_files = [f for f in files_in_folder if "e4" in f.lower() or "zww" in f.lower()]
            
            if zww_files:
                for zww_file in zww_files:
                    file_path = os.path.join(folder_full_path, zww_file)
                    print(f"Processing file: {zww_file}")
                    
                    try:
                        results = read_results_e4(file_path)
                        
                        # Print results for verification
                        print(f"Results from {zww_file}:")
                        for person_id, (timestamps, labels) in results.items():
                            print(f"  Person {person_id}:")
                            print(f"    Timestamps: {timestamps}")
                            print(f"    Labels: {labels}")
                            
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
            else:
                print(f"No files containing 'e4' or 'zww' found in folder: {folder}")
                
        except Exception as e:
            print(f"Error accessing folder {folder}: {e}")
    
    print("\nFinished processing all folders.") 