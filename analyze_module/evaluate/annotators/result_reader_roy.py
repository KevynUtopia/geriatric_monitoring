import json
from typing import Dict, Tuple, List
import os

def read_results_e5(file_path: str) -> Dict[str, Tuple[List[int], List[int]]]:  # Anonymized: e5=roy
    """
    Read the results JSON file and convert it into a dictionary with timestamps and states.
    
    Args:
        file_path (str): Path to the JSON file containing the results
        
    Returns:
        Dict[str, Tuple[List[int], List[int]]]: Dictionary where:
            - Keys are the person IDs (e.g., 'p_1', 'p_2', etc.)
            - Values are tuples containing:
                - List of timestamps (6-digit integers)
                - List of states (0 for normal, 1 for anomaly)
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
            # Find files containing "e5" or "roy" in their names (e5=roy anonymized)
            roy_files = [f for f in files_in_folder if "e5" in f.lower() or "roy" in f.lower()]
            
            if roy_files:
                for roy_file in roy_files:
                    file_path = os.path.join(folder_full_path, roy_file)
                    print(f"Processing file: {roy_file}")
                    
                    try:
                        results = read_results_e5(file_path)
                        
                        # Print results for verification
                        print(f"Results from {roy_file}:")
                        for person_id, (timestamps, labels) in results.items():
                            print(f"  Person {person_id}:")
                            print(f"    Timestamps: {timestamps}")
                            print(f"    Labels: {labels}")
                            
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
            else:
                print(f"No files containing 'e5' or 'roy' found in folder: {folder}")
                
        except Exception as e:
            print(f"Error accessing folder {folder}: {e}")
    
    print("\nFinished processing all folders.") 