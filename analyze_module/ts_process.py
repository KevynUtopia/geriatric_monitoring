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
        
        # For each gap, handle based on gap size
        for gap_idx in gaps:
            gap_start = time_seconds.iloc[gap_idx-1]
            gap_end = time_seconds.iloc[gap_idx]
            gap_size = gap_end - gap_start
            
            # Get the values before and after the gap for interpolation
            value_before = df[column].iloc[gap_idx-1]
            value_after = df[column].iloc[gap_idx]
            
            # Calculate number of 4-second intervals needed
            num_intervals = int(gap_size / 4)
            
            # Create new timestamps
            new_times = [gap_start + i*4 for i in range(1, num_intervals)]
            
            # Determine values based on gap size
            if gap_size > 300:
                # Fill with zeros for large gaps
                new_values = [0] * (num_intervals - 1)
            else:
                # Interpolate for smaller gaps
                new_values = []
                for i in range(1, num_intervals):
                    # Linear interpolation
                    t = gap_start + i*4
                    # Calculate interpolation weight
                    weight = (t - gap_start) / gap_size
                    interpolated_value = value_before * (1 - weight) + value_after * weight
                    new_values.append(interpolated_value)
            
            # Insert new values
            for t, v in zip(new_times, new_values):
                # Find the position to insert
                insert_pos = processed_df[time_col].searchsorted(t)
                processed_df.loc[insert_pos] = [t] + [v] * (len(processed_df.columns) - 1)
                processed_df = processed_df.sort_values(by=time_col).reset_index(drop=True)
    
    return processed_df

def process_all_folders(root_path, output_path, version):
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Walk through all directories
    for root, dirs, files in os.walk(root_path):
        # Check each subdirectory in the current directory
        for dir_name in dirs:
            # Only process folders that start with 'recording'
            if dir_name.startswith('recording'):
                recording_dir = os.path.join(root, dir_name)
                
                # Check if this recording directory has an alignment_soft_v7 subfolder
                alignment_dir = os.path.join(recording_dir, f"alignment_soft_{version}")
                if os.path.exists(alignment_dir) and os.path.isdir(alignment_dir):
                    # Process all CSV files in this alignment directory
                    for file in os.listdir(alignment_dir):
                        if file.endswith('.csv'):
                            file_path = os.path.join(alignment_dir, file)
                            
                            # Get the original CSV filename without extension
                            csv_name = os.path.splitext(file)[0]
                            
                            # Create output filename with folder and CSV information
                            output_file = os.path.join(output_path, f"ts_{dir_name}_{csv_name}.csv")
                            
                            # Check if output file already exists
                            if os.path.exists(output_file):
                                print(f"Skipping {file_path} - output file {output_file} already exists")
                                continue
                            
                            try:
                                # Process the CSV file
                                processed_df = process_time_series(file_path)
                                
                                processed_df.to_csv(output_file, index=False)
                                print(f"Processed {file_path} -> {output_file}")
                                
                            except Exception as e:
                                print(f"Error processing {file_path}: {str(e)}")

def main():
    # Get the current directory
    version = "v7"
    current_dir = f"/Users/kevynzhang/Downloads/results_{version}_alignment_soft"
    
    # Set output directory
    output_dir = os.path.join(current_dir, "DATASET")
    
    # Process all folders  
    process_all_folders(current_dir, output_dir, version)
    print(f"Processing complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()
