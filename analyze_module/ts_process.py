import os
import pandas as pd

def hhmmss_to_seconds(hhmmss):
    """Convert HHMMSS format to seconds."""
    hhmmss = str(hhmmss).zfill(6)
    hours = int(hhmmss[:2])
    minutes = int(hhmmss[2:4])
    seconds = int(hhmmss[4:])
    return hours * 3600 + minutes * 60 + seconds

def process_time_series(csv_path):
    df = pd.read_csv(csv_path)
    time_col = df.columns[0]
    time_seconds = df[time_col].apply(hhmmss_to_seconds)
    time_intervals = time_seconds.diff().dropna()
    
    processed_df = pd.DataFrame()
    processed_df[time_col] = time_seconds
    
    for column in df.columns[1:]:
        processed_df[column] = df[column].copy()
        gaps = time_intervals[time_intervals > 8].index
        
        for gap_idx in gaps:
            gap_start = time_seconds.iloc[gap_idx-1]
            gap_end = time_seconds.iloc[gap_idx]
            gap_size = gap_end - gap_start
            
            value_before = df[column].iloc[gap_idx-1]
            value_after = df[column].iloc[gap_idx]
            num_intervals = int(gap_size / 4)
            new_times = [gap_start + i*4 for i in range(1, num_intervals)]
            
            if gap_size > 300:
                new_values = [0] * (num_intervals - 1)
            else:
                new_values = []
                for i in range(1, num_intervals):
                    t = gap_start + i*4
                    weight = (t - gap_start) / gap_size
                    interpolated_value = value_before * (1 - weight) + value_after * weight
                    new_values.append(interpolated_value)
            
            for t, v in zip(new_times, new_values):
                insert_pos = processed_df[time_col].searchsorted(t)
                processed_df.loc[insert_pos] = [t] + [v] * (len(processed_df.columns) - 1)
                processed_df = processed_df.sort_values(by=time_col).reset_index(drop=True)
    
    return processed_df

def process_all_folders(root_path, output_path, version):
    os.makedirs(output_path, exist_ok=True)
    
    for root, dirs, files in os.walk(root_path):
        for dir_name in dirs:
            if dir_name.startswith('recording'):
                recording_dir = os.path.join(root, dir_name)
                alignment_dir = os.path.join(recording_dir, f"alignment_soft_{version}")
                if os.path.exists(alignment_dir) and os.path.isdir(alignment_dir):
                    for file in os.listdir(alignment_dir):
                        if file.endswith('.csv'):
                            file_path = os.path.join(alignment_dir, file)
                            csv_name = os.path.splitext(file)[0]
                            output_file = os.path.join(output_path, f"ts_{dir_name}_{csv_name}.csv")
                            
                            if os.path.exists(output_file):
                                continue
                            
                            try:
                                processed_df = process_time_series(file_path)
                                processed_df.to_csv(output_file, index=False)
                            except Exception as e:
                                print(f"Error processing {file_path}: {str(e)}")

def main():
    version = "v7"
    current_dir = f"/Users/kevynzhang/Downloads/results_{version}_alignment_soft"
    output_dir = os.path.join(current_dir, "DATASET")
    process_all_folders(current_dir, output_dir, version)

if __name__ == "__main__":
    main()
