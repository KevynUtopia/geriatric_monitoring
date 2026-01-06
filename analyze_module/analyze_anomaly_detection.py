import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
from datetime import datetime
from integrate_system.system_enging import SystemEngine

# Configuration
MIN_CSV_LENGTH = 500  # Minimum number of rows required in CSV
OUTPUT_PATH = "path_to_your_analysis_root/SNH/alarm_result_output"  # Output directory for results
DEBUG_MODE = False  # Set to True to process only one file for debugging

def convert_timestamp_to_seconds(timestamp):
    """Convert timestamp from hhmmss format to seconds from start of day"""
    timestamp_str = str(timestamp).zfill(6)  # Ensure 6 digits with leading zeros
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
    
    # Create a new figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    
    # Use timestamps directly (already in seconds format)
    if len(timestamps) > 0:
        time_seconds = list(timestamps)
    else:
        time_seconds = list(range(len(anomaly_scores)))
    
    # Ensure we have the same length
    min_length = min(len(time_seconds), len(anomaly_scores), len(ad_results))
    time_seconds = time_seconds[:min_length]
    anomaly_scores = anomaly_scores[:min_length]
    ad_results = ad_results[:min_length]
    
    # Plot soft anomaly scores
    ax.plot(time_seconds, anomaly_scores, 
           color='#3498db', 
           linewidth=1.5, 
           alpha=0.8,
           label='Soft Anomaly Scores')
    
    # Plot binary anomaly detection results
    ax.plot(time_seconds, ad_results, 
           color='#e74c3c', 
           linewidth=2, 
           alpha=0.9,
           drawstyle='steps-post',
           label='Binary Anomaly Detection')
    
    # Fill area under the binary results with transparency
    ax.fill_between(time_seconds, ad_results, 
                   color='#e74c3c', alpha=0.3,
                   step='post')
    
    # Mark binary anomaly points
    anomaly_indices = np.where(np.array(ad_results) > 0)[0]
    if len(anomaly_indices) > 0:
        anomaly_times = [time_seconds[i] for i in anomaly_indices]
        anomaly_values = [ad_results[i] for i in anomaly_indices]
        ax.scatter(anomaly_times, anomaly_values, 
                  color='#c0392b', 
                  s=30, zorder=5, alpha=0.8,
                  label=f'Binary Anomalies ({len(anomaly_indices)} detected)')
    
    # Formatting
    ax.set_title(f'Anomaly Detection Results: {filename}', fontweight='bold', fontsize=14)
    ax.set_xlabel('Time (HH:MM:SS)', fontsize=12)
    ax.set_ylabel('Anomaly Score', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    
    # Add horizontal lines at 0 and 1 for reference
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.5)
    
    # Format x-axis to show time in hhmmss format
    if len(time_seconds) > 0:
        time_range = time_seconds[-1] - time_seconds[0]
        if time_range > 7200:  # More than 2 hours
            tick_interval = 3600
        elif time_range > 3600:  # More than 1 hour
            tick_interval = 1800
        elif time_range > 1800:  # More than 30 minutes
            tick_interval = 900
        elif time_range > 900:  # More than 15 minutes
            tick_interval = 300
        else:
            tick_interval = 60
        
        # Generate tick positions
        start_tick = time_seconds[0] - (time_seconds[0] % tick_interval)
        if start_tick < time_seconds[0]:
            start_tick += tick_interval
        
        tick_positions = list(range(int(start_tick), int(time_seconds[-1]) + tick_interval, tick_interval))
        tick_positions = [t for t in tick_positions if time_seconds[0] <= t <= time_seconds[-1]]
        
        # Create labels in hhmmss format
        tick_labels = [convert_seconds_to_timestamp(t) for t in tick_positions]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45)
    
    plt.tight_layout()
    
    # Save the plot with the same name as original file
    plot_output_path = os.path.join(output_dir, f'{filename}.svg')
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved anomaly detection plot: {plot_output_path}")
    
    plt.close()  # Close the figure to free memory

def process_csv_file(csv_path, output_dir):
    """Process a single CSV file with the SystemEngine"""
    
    filename = os.path.basename(csv_path).replace('.csv', '')
    print(f"Processing CSV: {filename}")
    
    try:
        # Check CSV length first
        df_check = pd.read_csv(csv_path)
        if len(df_check) < MIN_CSV_LENGTH:
            print(f"Skipping {filename}: Only {len(df_check)} rows (minimum {MIN_CSV_LENGTH} required)")
            return False
        
        print(f"CSV has {len(df_check)} rows, proceeding with analysis...")
        
        # Create SystemEngine instance
        engine = SystemEngine()
        
        # Process data with the engine
        engine.process_data(csv_path)
        
        # Get the processed dataframe and anomaly detection results
        df = engine.get_dataframe()
        
        # Convert data to tensor for anomaly detection
        data = torch.tensor(df.values, dtype=torch.float32)
        
        # Perform anomaly detection
        ad_results, anomaly_scores = engine.anomaly_detection(data)
        
        # Get timestamps from original data
        original_timestamps = df_check['timestamp'].values if 'timestamp' in df_check.columns else []
        
        # Plot and save results
        plot_anomaly_detection_results(csv_path, ad_results, anomaly_scores, original_timestamps, output_dir, filename)
        
        # Print summary statistics
        total_anomalies = np.sum(anomaly_scores)
        anomaly_rate = total_anomalies / len(anomaly_scores) * 100
        print(f"Analysis complete for {filename}: {total_anomalies} anomalies detected ({anomaly_rate:.2f}% anomaly rate)")
        
        return True
        
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_directory(directory_path):
    """Process all CSV files in the specified directory"""
    
    if not os.path.exists(directory_path):
        print(f"Error: Directory does not exist: {directory_path}")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Find all CSV files in the directory
    csv_pattern = os.path.join(directory_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files in {directory_path}")
    
    processed_count = 0
    for csv_file in sorted(csv_files):
        # specify a csv file as debug mode
        if DEBUG_MODE:
            csv_file = 'path_to_your_analysis_root/SNH/TSAD_DATASET/ts_recording_2019_06_22_9_20_am_p_5.csv'
        # skip those csv_file with "giver" in the name
        if "giver" in csv_file:
            continue
        if process_csv_file(csv_file, OUTPUT_PATH):
            processed_count += 1
            
        # Debug mode: only process one file
        if DEBUG_MODE:
            print("Debug mode: stopping after first file")
            break
    
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"Total CSV files found: {len(csv_files)}")
    print(f"Total CSV files processed: {processed_count}")
    print(f"Results saved to: {OUTPUT_PATH}")
    print(f"{'='*80}")

def main():
    # Base path containing CSV files to process
    base_path = "path_to_your_analysis_root/SNH/TSAD_DATASET"
    
    print("Starting Anomaly Detection Analysis")
    print(f"Input directory: {base_path}")
    print(f"Output directory: {OUTPUT_PATH}")
    print(f"Minimum CSV length: {MIN_CSV_LENGTH}")
    print(f"Debug mode: {DEBUG_MODE}")
    print()
    
    # Process all CSV files in the directory
    process_directory(base_path)
    
    print("\nAnomaly detection analysis complete!")

if __name__ == "__main__":
    main() 