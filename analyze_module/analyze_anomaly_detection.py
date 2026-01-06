import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
from integrate_system.system_enging import SystemEngine

MIN_CSV_LENGTH = 500
OUTPUT_PATH = "path_to_your_analysis_root/SNH/alarm_result_output"
DEBUG_MODE = False

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
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def plot_anomaly_detection_results(csv_path, ad_results, anomaly_scores, timestamps, output_dir, filename):
    """Plot anomaly detection results and save to file"""
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    
    if len(timestamps) > 0:
        time_seconds = list(timestamps)
    else:
        time_seconds = list(range(len(anomaly_scores)))
    
    min_length = min(len(time_seconds), len(anomaly_scores), len(ad_results))
    time_seconds = time_seconds[:min_length]
    anomaly_scores = anomaly_scores[:min_length]
    ad_results = ad_results[:min_length]
    
    ax.plot(time_seconds, anomaly_scores, color='#3498db', linewidth=1.5, alpha=0.8, label='Soft Anomaly Scores')
    ax.plot(time_seconds, ad_results, color='#e74c3c', linewidth=2, alpha=0.9, drawstyle='steps-post', label='Binary Anomaly Detection')
    ax.fill_between(time_seconds, ad_results, color='#e74c3c', alpha=0.3, step='post')
    
    anomaly_indices = np.where(np.array(ad_results) > 0)[0]
    if len(anomaly_indices) > 0:
        anomaly_times = [time_seconds[i] for i in anomaly_indices]
        anomaly_values = [ad_results[i] for i in anomaly_indices]
        ax.scatter(anomaly_times, anomaly_values, color='#c0392b', s=30, zorder=5, alpha=0.8, label=f'Binary Anomalies ({len(anomaly_indices)} detected)')
    
    ax.set_title(f'Anomaly Detection Results: {filename}', fontweight='bold', fontsize=14)
    ax.set_xlabel('Time (HH:MM:SS)', fontsize=12)
    ax.set_ylabel('Anomaly Score', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.5)
    
    if len(time_seconds) > 0:
        time_range = time_seconds[-1] - time_seconds[0]
        tick_interval = 3600 if time_range > 7200 else 1800 if time_range > 3600 else 900 if time_range > 1800 else 300 if time_range > 900 else 60
        
        start_tick = time_seconds[0] - (time_seconds[0] % tick_interval)
        if start_tick < time_seconds[0]:
            start_tick += tick_interval
        
        tick_positions = list(range(int(start_tick), int(time_seconds[-1]) + tick_interval, tick_interval))
        tick_positions = [t for t in tick_positions if time_seconds[0] <= t <= time_seconds[-1]]
        tick_labels = [convert_seconds_to_timestamp(t) for t in tick_positions]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45)
    
    plt.tight_layout()
    plot_output_path = os.path.join(output_dir, f'{filename}.svg')
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_csv_file(csv_path, output_dir):
    """Process a single CSV file with the SystemEngine"""
    filename = os.path.basename(csv_path).replace('.csv', '')
    
    try:
        df_check = pd.read_csv(csv_path)
        if len(df_check) < MIN_CSV_LENGTH:
            return False
        
        engine = SystemEngine()
        engine.process_data(csv_path)
        df = engine.get_dataframe()
        data = torch.tensor(df.values, dtype=torch.float32)
        ad_results, anomaly_scores = engine.anomaly_detection(data)
        
        original_timestamps = df_check['timestamp'].values if 'timestamp' in df_check.columns else []
        plot_anomaly_detection_results(csv_path, ad_results, anomaly_scores, original_timestamps, output_dir, filename)
        
        return True
        
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        return False

def process_directory(directory_path):
    """Process all CSV files in the specified directory"""
    if not os.path.exists(directory_path):
        print(f"Error: Directory does not exist: {directory_path}")
        return
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    csv_pattern = os.path.join(directory_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        return
    
    processed_count = 0
    for csv_file in sorted(csv_files):
        if DEBUG_MODE:
            csv_file = 'path_to_your_analysis_root/SNH/TSAD_DATASET/ts_recording_2019_06_22_9_20_am_p_5.csv'
        if "giver" in csv_file:
            continue
        if process_csv_file(csv_file, OUTPUT_PATH):
            processed_count += 1
        if DEBUG_MODE:
            break
    
    print(f"Processed {processed_count}/{len(csv_files)} files. Results saved to: {OUTPUT_PATH}")

def main():
    base_path = "path_to_your_analysis_root/SNH/TSAD_DATASET"
    process_directory(base_path)

if __name__ == "__main__":
    main() 