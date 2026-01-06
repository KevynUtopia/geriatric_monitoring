import re
import os
from typing import Dict, Tuple, List

def read_results_e2(file_path: str) -> Dict[str, Tuple[List[int], List[int]]]:  # Anonymized: e2=zzr
    """
    Read the results TXT file and convert it into a dictionary with timestamps and states.
    
    Args:
        file_path (str): Path to the TXT file containing the results
        
    Returns:
        Dict[str, Tuple[List[int], List[int]]]: Dictionary where:
            - Keys are the person IDs (e.g., 'p_4', 'p_7', etc.)
            - Values are tuples containing:
                - List of timestamps (6-digit integers)
                - List of states (0 for normal, 1 for anomaly)
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
            # Find files containing "e2" or "zzr" in their names (e2=zzr anonymized)
            zzr_files = [f for f in files_in_folder if "e2" in f.lower() or "zzr" in f.lower()]
            if zzr_files:
                for zzr_file in zzr_files:
                    file_path = os.path.join(folder_full_path, zzr_file)
                    print(f"Processing file: {zzr_file}")
                    
                    try:
                        results = read_results_e2(file_path)
                        
                        # Print results for verification
                        print(f"Results from {zzr_file}:")
                        for person_id, (timestamps, labels) in results.items():
                            print(f"  Person {person_id}:")
                            print(f"    Timestamps: {timestamps}")
                            print(f"    Labels: {labels}")
                            
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
            else:
                print(f"No files containing 'e2' or 'zzr' found in folder: {folder}")
                
        except Exception as e:
            print(f"Error accessing folder {folder}: {e}")
    
    print("\nFinished processing all folders.") 