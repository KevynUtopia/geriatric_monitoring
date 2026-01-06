import os
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import shutil

def hhmmss_to_seconds(hhmmss):
    """Convert HHMMSS format to seconds."""
    hhmmss = str(hhmmss).zfill(6)  # Ensure 6 digits with leading zeros
    hours = int(hhmmss[:2])
    minutes = int(hhmmss[2:4])
    seconds = int(hhmmss[4:])
    return hours * 3600 + minutes * 60 + seconds

def process_time_series(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get the timestamp column (first column)
    time_col = df.columns[0]
    
    # Convert time to seconds for plotting
    time_seconds = df[time_col].apply(hhmmss_to_seconds)
    
    # Calculate time intervals
    time_intervals = time_seconds.diff().dropna()
    
    # Create a new DataFrame with processed values
    processed_df = pd.DataFrame()
    processed_df[time_col] = time_seconds
    
    # Process each measurement column
    for column in df.columns[1:]:
        # Keep original values
        processed_df[column] = df[column].copy()
        
        # Find gaps longer than 8 seconds
        gaps = time_intervals[time_intervals > 8].index
        
        # For each gap, insert zeros with 4-second intervals
        for gap_idx in gaps:
            gap_start = time_seconds.iloc[gap_idx-1]
            gap_end = time_seconds.iloc[gap_idx]
            
            # Calculate number of 4-second intervals needed
            num_intervals = int((gap_end - gap_start) / 4)
            
            # Create new timestamps and zero values
            new_times = [gap_start + i*4 for i in range(1, num_intervals)]
            new_values = [0] * (num_intervals - 1)
            
            # Insert new values
            for t, v in zip(new_times, new_values):
                # Find the position to insert
                insert_pos = processed_df[time_col].searchsorted(t)
                processed_df.loc[insert_pos] = [t] + [v] * (len(processed_df.columns) - 1)
                processed_df = processed_df.sort_values(by=time_col).reset_index(drop=True)
    
    return processed_df

def process_all_folders(root_path, output_path):
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Walk through all directories
    for root, dirs, files in os.walk(root_path):
        # Check if current directory contains "alignment_soft_v6"
        if "alignment_soft_v7" in root:
            # Get the parent folder name (one level up from alignment_soft_v6)
            parent_folder = os.path.basename(os.path.dirname(root))
            
            # Process all CSV files in this directory
            for file in files:
                if file.endswith('.csv'):
                    input_file = os.path.join(root, file)
                    try:
                        # Process the CSV file
                        processed_df = process_time_series(input_file)
                        
                        # Get the original CSV filename without extension
                        csv_name = os.path.splitext(file)[0]
                        
                        # Create output filename with folder and CSV information
                        output_file = os.path.join(output_path, f"ts_{parent_folder}_{csv_name}.csv")
                        processed_df.to_csv(output_file, index=False)
                        print(f"Processed {input_file} -> {output_file}")
                        
                    except Exception as e:
                        print(f"Error processing {input_file}: {str(e)}")

def main():
    # Get the current directory
    current_dir = "path_to_your_root/results_v7_analyze_soft"
    
    # Set output directory
    output_dir = os.path.join(current_dir, "OUTPUT")
    
    # Process all folders
    process_all_folders(current_dir, output_dir)
    print(f"Processing complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()
