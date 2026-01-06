import json
from typing import Dict, Tuple, List
import os

def read_results_e1(file_path: str) -> Dict[str, Tuple[List[int], List[int]]]:  # Anonymized: e1=cyy
    """
    Read the results JSON file and convert it into a dictionary with timestamps and anomaly labels.
    
    Args:
        file_path (str): Path to the JSON file containing the results
        
    Returns:
        Dict[str, Tuple[List[int], List[int]]]: Dictionary where:
            - Keys are the person IDs (e.g., 'p_4', 'p_7', etc.)
            - Values are tuples containing:
                - List of timestamps (6-digit integer)
                - List of anomaly labels (1 for anomaly, 0 for normal)
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
            # Find files containing "e1" or "cyy" in their names (e1=cyy anonymized)
            cyy_files = [f for f in files_in_folder if "e1" in f.lower() or "cyy" in f.lower()]
            
            if cyy_files:
                for cyy_file in cyy_files:
                    file_path = os.path.join(folder_full_path, cyy_file)
                    print(f"Processing file: {cyy_file}")
                    
                    try:
                        results = read_results_e1(file_path)
                        
                        # Print results for verification
                        print(f"Results from {cyy_file}:")
                        for person_id, (timestamps, labels) in results.items():
                            print(f"  Person {person_id}:")
                            print(f"    Timestamps: {timestamps}")
                            print(f"    Labels: {labels}")
                            
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
            else:
                print(f"No files containing 'e1' or 'cyy' found in folder: {folder}")
                
        except Exception as e:
            print(f"Error accessing folder {folder}: {e}")
    
    print("\nFinished processing all folders.")