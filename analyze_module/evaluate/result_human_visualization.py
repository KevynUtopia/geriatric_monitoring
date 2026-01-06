import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob
import pandas as pd
import argparse

# Configuration variables - modify these to change behavior
COLOR_PALETTE = 'seaborn'  # Choose from: tableau, pastel, viridis, seaborn, bright, dark, cool
DEBUG_MODE = False  # Set to True to process only one file for debugging
SHOW_PALETTES = False  # Set to True to show available palettes and exit

# Define multiple color palettes for visualization
COLOR_PALETTES = {
    'tableau': ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
                '#59a14f', '#edc948', '#b07aa1', '#ff9da7'],
    'pastel': ['#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b',
               '#d0bbff', '#debb9b', '#fab0e4', '#cfcfcf'],
    'viridis': ['#003f5c', '#2f4b7c', '#665191', '#a05195',
                '#d45087', '#f95d6a', '#ff7c43', '#ffa600'],
    'seaborn': ['#4c72b0', '#dd8452', '#55a868', '#c44e52',
                '#8172b3', '#937860', '#da8bc3', '#8c8c8c'],
    'bright': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
               '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'],
    'dark': ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71',
             '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22'],
    'cool': ['#6C5CE7', '#74B9FF', '#00B894', '#FDCB6E',
             '#E17055', '#FD79A8', '#636E72', '#00CEC9']
}


START_TIME = "100000"
END_TIME = "120000"
# START_TIME = None
# END_TIME = None

# Optional: overlay system alarm CSVs from a parallel folder structure
# Provide the base folder; CSVs containing 'seperate' in filename will be ignored
EXTERNAL_ALARM_BASE = \
    'path_to_your_analysis_root/SNH/0_SYSTEM_RESULTS/alarm_result_output/alarm_result_output_v10'
def _detect_time_value_to_seconds(value):
    """Convert a single timestamp value to seconds. Supports hhmmss strings/ints, HH:MM:SS, or raw seconds."""
    try:
        s = str(value).strip()
        if not s:
            return None
        # HHMMSS (6 digits)
        if s.isdigit() and len(s) == 6:
            hours = int(s[:2])
            minutes = int(s[2:4])
            seconds = int(s[4:6])
            return hours * 3600 + minutes * 60 + seconds
        # HH:MM:SS
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
        # Plain seconds
        if s.replace('.', '', 1).isdigit():
            return int(float(s))
    except Exception:
        pass
    return None

def _read_alarm_csv_series(csv_path):
    """Read a CSV, find time and 'alarm' columns, return (times_seconds_sorted, values) lists.
    Ignores files containing 'seperate' in the filename.
    """
    basename = os.path.basename(csv_path)
    if "seperate" in basename:
        return [], []
    time_candidates = ["time", "timestamp", "second", "seconds", "Time", "Second"]
    alarm_candidates = ["alarm", "Alarm", "ALARM"]
    import csv
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            time_col = None
            for c in time_candidates:
                if c in headers:
                    time_col = c
                    break
            alarm_col = None
            for c in alarm_candidates:
                if c in headers:
                    alarm_col = c
                    break
            if time_col is None or alarm_col is None:
                return [], []
            time_values = []
            alarm_values = []
            for row in reader:
                sec = _detect_time_value_to_seconds(row.get(time_col))
                if sec is None:
                    continue
                try:
                    alarm_val_raw = row.get(alarm_col)
                    alarm_val = int(float(alarm_val_raw)) if alarm_val_raw is not None and str(alarm_val_raw) != "" else 0
                except Exception:
                    alarm_val = 0
                time_values.append(sec)
                alarm_values.append(alarm_val)
            combined = sorted(zip(time_values, alarm_values))
            if not combined:
                return [], []
            times_sorted, values_sorted = zip(*combined)
            return list(times_sorted), list(values_sorted)
    except Exception:
        return [], []

def _discover_recording_alarm_csvs(external_base_dir, recording_stem):
    """Find CSV files under external_base_dir that relate to the given recording stem and don't contain 'seperate'."""
    if external_base_dir is None or not os.path.isdir(external_base_dir):
        return []
    pattern = os.path.join(external_base_dir, "**", "*.csv")
    try:
        all_csvs = glob.glob(pattern, recursive=True)
        return [p for p in all_csvs if "seperate" not in os.path.basename(p) and recording_stem in os.path.basename(p)]
    except Exception:
        return []

def _build_aggregated_alarm_series(external_base_dir, recording_stem, person_key, start_time, end_time, postfix='timemixer'):
    """Aggregate multiple alarm CSVs by taking max across sources per second, forward-filled across [start_time, end_time]."""
    if external_base_dir is None or not os.path.isdir(external_base_dir):
        return [], [], []
    csv_path = os.path.join(external_base_dir, f"ts_{recording_stem}_{person_key}_{postfix}.csv")
    if not os.path.exists(csv_path):
        return [], [], []
    # read csv_path and take the column of 'alarm' 'timestamp' and 'seconds'
    df = pd.read_csv(csv_path)
    aligned_times = df['timestamp'].values
    seconds = df['seconds'].values
    values = df['alarm'].values
    # clip aligned_times and corresponding values to start_time and end_time
    clip_index = (seconds >= start_time) & (seconds <= end_time)
    aligned_times = aligned_times[clip_index]
    values = values[clip_index]
    seconds = seconds[clip_index]
    return aligned_times, values, seconds

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

def forward_fill_interpolation(timestamps, labels, start_time=None, end_time=None):
    """
    Forward fill interpolation between timestamps
    Returns arrays for continuous plotting
    """
    if not timestamps or not labels:
        return [], []
    
    # Convert timestamps to seconds
    time_seconds = [convert_timestamp_to_seconds(ts) for ts in timestamps]
    
    # Determine the range
    if start_time is None:
        start_time = min(time_seconds)
    if end_time is None:
        end_time = max(time_seconds)
    
    # Create continuous time array
    continuous_time = list(range(start_time, end_time + 1))
    continuous_labels = []
    
    current_label = 0  # Default starting label
    ts_index = 0
    
    for t in continuous_time:
        # Check if we need to update the label
        while ts_index < len(time_seconds) and time_seconds[ts_index] <= t:
            current_label = labels[ts_index]
            ts_index += 1
        continuous_labels.append(current_label)
    
    return continuous_time, continuous_labels

def forward_fill_interpolation_seconds(timestamps_seconds, labels, start_time=None, end_time=None):
    """
    Forward fill interpolation between timestamps (already in seconds)
    Returns arrays for continuous plotting
    """
    if not timestamps_seconds or not labels:
        return [], []
    
    # Determine the range
    if start_time is None:
        start_time = min(timestamps_seconds)
    if end_time is None:
        end_time = max(timestamps_seconds)
    
    # Create continuous time array
    continuous_time = list(range(start_time, end_time + 1))
    continuous_labels = []
    
    current_label = 0  # Default starting label
    ts_index = 0
    
    for t in continuous_time:
        # Check if we need to update the label
        while ts_index < len(timestamps_seconds) and timestamps_seconds[ts_index] <= t:
            current_label = labels[ts_index]
            ts_index += 1
        continuous_labels.append(current_label)
    
    return continuous_time, continuous_labels

def _recurrence_plot(signal, dim=7, tau=10, epsilon=None):
    """Compute recurrence plot boolean matrix for a 1D signal.
    - dim: embedding dimension
    - tau: time delay
    - epsilon: threshold; if None, use 10th percentile of distances
    """
    n = len(signal) - (dim - 1) * tau
    if n <= 0:
        raise ValueError("Signal too short for phase space reconstruction")
    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = np.asarray(signal)[i * tau : i * tau + n]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(embedded[i] - embedded[j])
    if epsilon is None:
        epsilon = np.percentile(distances, 10)
    rp = distances <= epsilon
    return rp

def _plot_and_save_recurrence(person_key, system_series, output_dir):
    """Plot and save recurrence plot image for a given anomaly score sequence."""
    try:
        if system_series is None or len(system_series) == 0:
            return
        rp = _recurrence_plot(system_series, dim=7, tau=10, epsilon=None)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(rp, cmap='binary', origin='lower', aspect='auto')
        # ax.set_title(f'Recurrence Plot - {person_key}', fontsize=30, fontweight='bold')s
        ax.set_xlabel('Time index', fontsize=30)
        ax.set_ylabel('Time index', fontsize=30)
        ax.tick_params(axis='both', labelsize=30)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontweight('bold')
        plt.tight_layout()
        rp_path = os.path.join(output_dir, f'{person_key}_rp.svg')
        plt.savefig(rp_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved recurrence plot: {rp_path}")
    except Exception as e:
        print(f"Warning: Failed to save recurrence plot for {person_key}: {e}")

def plot_single_person(person_key, evaluator_data, global_start, global_end, output_dir, filename, color_palette='tableau', plot_RP=False):
    """Plot results for a single person from all evaluators as a separate figure,
    but only plot results in the middle of START_TIME and END_TIME (hhmmss strings).
    If no results in the middle, skip this person (return False).
    """
    # Check if time filtering is enabled
    if START_TIME is None or END_TIME is None:
        # No time filtering - use all data
        start_sec = global_start
        end_sec = global_end
        use_time_filtering = False
    else:
        # Convert START_TIME and END_TIME to seconds
        start_sec = convert_timestamp_to_seconds(START_TIME)
        end_sec = convert_timestamp_to_seconds(END_TIME)
        use_time_filtering = True

    # Prepare to collect filtered data for all evaluators
    filtered_evaluator_data = []

    for i, (evaluator_name, evaluator_info) in enumerate(evaluator_data.items()):
        results = evaluator_info['results']
        cam = evaluator_info['cam']
        if person_key in results:
            p_data = results[person_key]
            if 'timestamps' in p_data and 'labels' in p_data:
                timestamps = p_data['timestamps']
                labels = p_data['labels']
                if timestamps and labels:
                    # Convert all timestamps to seconds for filtering
                    ts_sec = [convert_timestamp_to_seconds(ts) for ts in timestamps]
                    
                    if use_time_filtering:
                        # Find the range of data that includes the specified interval
                        # We need to include data before start_sec to maintain proper forward fill
                        # and data after end_sec to handle area filling correctly
                        
                        # Find the first timestamp that's <= start_sec (for proper forward fill)
                        start_idx = 0
                        for idx, t in enumerate(ts_sec):
                            if t <= start_sec:
                                start_idx = idx
                            else:
                                break
                        
                        # Find the last timestamp that's >= end_sec (for proper area filling)
                        end_idx = len(ts_sec) - 1
                        for idx in range(len(ts_sec) - 1, -1, -1):
                            if ts_sec[idx] >= end_sec:
                                end_idx = idx
                            else:
                                break
                        
                        # Extract the extended range of data
                        extended_timestamps = ts_sec[start_idx:end_idx + 1]
                        extended_labels = labels[start_idx:end_idx + 1]
                        
                        # Check if there's any data within the specified interval
                        has_data_in_interval = any(start_sec <= t <= end_sec for t in extended_timestamps)
                        
                        if has_data_in_interval and extended_timestamps:
                            filtered_evaluator_data.append({
                                'evaluator_name': evaluator_name,
                                'cam': cam,
                                'color_index': i,
                                'timestamps': extended_timestamps,
                                'labels': extended_labels
                            })
                    else:
                        # No time filtering - use all data
                        filtered_evaluator_data.append({
                            'evaluator_name': evaluator_name,
                            'cam': cam,
                            'color_index': i,
                            'timestamps': ts_sec,
                            'labels': labels
                        })

    # If no data for this person, skip (return False)
    if not filtered_evaluator_data:
        return False

    # Create a new figure for this person
    fig, ax = plt.subplots(1, 1, figsize=(20.0, 12))
    ax.set_ylabel('Human Opinion', fontsize=40, fontweight='bold')
    ax2 = None
    ax2 = None

    # Get the selected color palette
    if color_palette not in COLOR_PALETTES:
        print(f"Warning: Color palette '{color_palette}' not found. Using 'tableau' instead.")
        color_palette = 'tableau'
    selected_colors = COLOR_PALETTES[color_palette]

    # Track if any data was plotted for this person
    data_plotted = False
    
    # Plot data from each evaluator for this person (only filtered data)
    for human_idx, entry in enumerate(filtered_evaluator_data):
        evaluator_name = entry['evaluator_name']
        cam = entry['cam']
        color_index = entry['color_index'] % len(selected_colors)
        color = selected_colors[color_index]
        timestamps = entry['timestamps']
        labels = entry['labels']

        # Forward fill interpolation within the extended range
        cont_time, cont_labels = forward_fill_interpolation_seconds(
            timestamps, labels, 
            start_time=min(timestamps), end_time=max(timestamps)
        )

        if cont_time and cont_labels:
            # Plot the complete line first (for proper area filling)
            ax.plot(cont_time, cont_labels,
                    color=color,
                    linewidth=2,
                    label=f'human{human_idx + 1}',
                    alpha=0.9)

            # Fill area under the curve with transparency
            ax.fill_between(cont_time, cont_labels,
                            color=color, alpha=0.3)

            # Mark original timestamps
            if use_time_filtering:
                # Filter timestamps to only show those within the interval
                filtered_orig_times = []
                filtered_orig_labels = []
                for t, l in zip(timestamps, labels):
                    if start_sec <= t <= end_sec:
                        filtered_orig_times.append(t)
                        filtered_orig_labels.append(l)
                
                if filtered_orig_times:
                    ax.scatter(filtered_orig_times, filtered_orig_labels,
                               color=color,
                               s=50, zorder=5, alpha=1.0,
                               edgecolors='white', linewidths=1)
            else:
                # Show all original timestamps
                ax.scatter(timestamps, labels,
                           color=color,
                           s=50, zorder=5, alpha=1.0,
                           edgecolors='white', linewidths=1)
            
            data_plotted = True

    # Overlay external system alarm series on secondary y-axis if configured
    if EXTERNAL_ALARM_BASE is not None:
        recording_stem = filename  # e.g., recording_2019_06_22_9_20_am
        alarm_times, alarm_values, alarm_seconds = _build_aggregated_alarm_series(
            EXTERNAL_ALARM_BASE, recording_stem, person_key, start_sec, end_sec
        )
        if alarm_times is not None:
            max_val = max(alarm_values) if len(alarm_values) > 0 else 1
            if max_val > 0 and max_val != 1:
                alarm_plot_values = [v / max_val for v in alarm_values]
            else:
                alarm_plot_values = alarm_values
            # Interpolate system series to the same per-second grid as human x-axis
            full_time = list(range(start_sec, end_sec + 1))
            if len(alarm_seconds) > 1:
                aligned_system = np.interp(full_time, alarm_seconds, alarm_plot_values)
            elif len(alarm_seconds) == 1:
                aligned_system = [alarm_plot_values[0]] * len(full_time)
            else:
                aligned_system = []
            if len(aligned_system) > 0:
                ax2 = ax.twinx()
                ax2.set_ylim(-0.05, 1.05)
                ax2.set_ylabel('Anomaly Score', color='black', fontsize=40, fontweight='bold')
                ax2.plot(full_time, aligned_system, color='black', linewidth=2.0, label='system', zorder=4)
                ax2.tick_params(axis='y', labelcolor='black', labelsize=40)
                for t in ax2.get_yticklabels():
                    t.set_fontweight('bold')

    # Formatting
    ax.grid(True, alpha=0.3)
    # Merge legends from both axes if system overlay exists
    if ax2 is not None:
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right', framealpha=0.9, prop={'size': 40, 'weight': 'bold'})
    else:
        ax.legend(loc='upper right', framealpha=0.9, prop={'size': 40, 'weight': 'bold'})
    ax.set_ylim(-0.05, 1.05)
    
    # Set x-axis limits to only show the specified interval
    ax.set_xlim(start_sec, end_sec)

    # Add horizontal lines at 0 and 1 for reference
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.5)

    # Configure x-axis ticks every 30 minutes (1800 seconds) and format as hhmmss
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1800))
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: convert_seconds_to_timestamp(int(x)).replace(':', '')))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    # Set axis tick font sizes and weight
    ax.tick_params(axis='both', labelsize=40)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight('bold')
    # Hide y tick labels for left axis to keep the clean look
    ax.set_yticklabels([])

    plt.tight_layout()

    # Save the individual plot
    person_output_path = os.path.join(output_dir, f'{person_key}.svg')
    plt.savefig(person_output_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved plot: {person_output_path}")
    
    plt.close()  # Close the figure to free memory
    # Save aligned time series to CSV for this person

    try:
        full_time = list(range(start_sec, end_sec + 1))
        series_dict = {"time_seconds": full_time}

        for human_idx, entry in enumerate(filtered_evaluator_data):
            col_name = f"human{human_idx + 1}"
            timestamps = entry['timestamps']
            labels = entry['labels']
            cont_time_full, cont_labels_full = forward_fill_interpolation_seconds(
                timestamps, labels, start_time=start_sec, end_time=end_sec
            )

            if cont_time_full and len(cont_time_full) == len(full_time):
                series_dict[col_name] = cont_labels_full
            else:
                value_by_time = {t: v for t, v in zip(cont_time_full, cont_labels_full)}
                series_dict[col_name] = [value_by_time.get(t, 0) for t in full_time]

        # Add system series to CSV if available
        system_series = None
        if EXTERNAL_ALARM_BASE is not None:
            alarm_times, alarm_values, alarm_seconds = _build_aggregated_alarm_series(
                EXTERNAL_ALARM_BASE, filename, person_key, start_sec, end_sec
            )
            if alarm_times is not None:
                max_val = max(alarm_values) if len(alarm_values) > 0 else 1
                if max_val > 0 and max_val != 1:
                    alarm_plot_values = [v / max_val for v in alarm_values]
                else:
                    alarm_plot_values = alarm_values
                if len(alarm_seconds) > 1:
                    system_series = np.interp(full_time, alarm_seconds, alarm_plot_values).tolist()
                elif len(alarm_seconds) == 1:
                    system_series = [alarm_plot_values[0]] * len(full_time)
                else:
                    system_series = []
                if system_series:
                    series_dict['system'] = system_series

        import csv
        csv_output_path = os.path.join(output_dir, f"{person_key}.csv")
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            headers = list(series_dict.keys())
            writer.writerow(headers)
            for i in range(len(full_time)):
                row = [series_dict[h][i] for h in headers]
                writer.writerow(row)
        print(f"Saved time series CSV: {csv_output_path}")
    except Exception as e:
        print(f"Warning: Failed to save CSV for {person_key}: {e}")
    
    # Also save recurrence plot for the anomaly (system) sequence if available
    try:
        if EXTERNAL_ALARM_BASE is not None:
            # Reconstruct aligned system sequence as above
            alarm_times, alarm_values, alarm_seconds = _build_aggregated_alarm_series(
                EXTERNAL_ALARM_BASE, filename, person_key, start_sec, end_sec
            )
            system_series_for_rp = None
            if alarm_times is not None:
                max_val = max(alarm_values) if len(alarm_values) > 0 else 1
                if max_val > 0 and max_val != 1:
                    alarm_plot_values = [v / max_val for v in alarm_values]
                else:
                    alarm_plot_values = alarm_values
                full_time = list(range(start_sec, end_sec + 1))
                if len(alarm_seconds) > 1:
                    system_series_for_rp = np.interp(full_time, alarm_seconds, alarm_plot_values).tolist()
                elif len(alarm_seconds) == 1:
                    system_series_for_rp = [alarm_plot_values[0]] * len(full_time)
                else:
                    system_series_for_rp = []


            if plot_RP:
                if system_series_for_rp is not None:
                    _plot_and_save_recurrence(person_key, system_series_for_rp, output_dir)
    except Exception as e:
        print(f"Warning: Failed RP generation for {person_key}: {e}")

    return data_plotted

def plot_summary_figure(evaluator_data, person_keys, global_start, global_end, output_dir, filename, color_palette='tableau', plot_RP=False):
    """Create a large summary plot with multiple subplots (3 rows x 6 columns)"""
    
    # Take only first 18 persons
    display_persons = person_keys[:18]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 6, figsize=(20, 12))
    axes = axes.flatten()  # Convert to 1D array for easier indexing
    
    # Get the selected color palette
    if color_palette not in COLOR_PALETTES:
        print(f"Warning: Color palette '{color_palette}' not found. Using 'tableau' instead.")
        color_palette = 'tableau'
    
    selected_colors = COLOR_PALETTES[color_palette]
    
    # Plot each person in a subplot
    for idx, person_key in enumerate(display_persons):
        ax = axes[idx]
        
        # Track if any data was plotted for this person
        data_plotted = False
        
        # Plot data from each evaluator for this person
        for i, (evaluator_name, evaluator_info) in enumerate(evaluator_data.items()):
            results = evaluator_info['results']
            cam = evaluator_info['cam']
            
            # Use color cycling if we have more evaluators than colors
            color_index = i % len(selected_colors)
            color = selected_colors[color_index]
            
            # Check if this evaluator has data for this person
            if person_key in results:
                p_data = results[person_key]
                
                if 'timestamps' in p_data and 'labels' in p_data:
                    timestamps = p_data['timestamps']
                    labels = p_data['labels']
                    
                    if timestamps and labels:
                        # Forward fill interpolation
                        cont_time, cont_labels = forward_fill_interpolation(
                            timestamps, labels, global_start, global_end
                        )
                        
                        # Plot the line
                        ax.plot(cont_time, cont_labels, 
                               color=color, 
                               linewidth=1.5, 
                               alpha=0.8)
                        
                        # Fill area under the curve with transparency
                        ax.fill_between(cont_time, cont_labels, 
                                       color=color, alpha=0.2)
                        
                        # Mark original timestamps (smaller markers for summary)
                        original_times = [convert_timestamp_to_seconds(ts) for ts in timestamps]
                        ax.scatter(original_times, labels, 
                                  color=color, 
                                  s=20, zorder=5, alpha=0.8,
                                  edgecolors='white', linewidths=0.5)
                        
                        data_plotted = True
        
        # Formatting for each subplot
        if data_plotted:
            ax.set_title(f'{person_key}', fontweight='bold', fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            
            # Add horizontal lines at 0 and 1 for reference
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
            
            # Remove axis labels as requested
            ax.set_xticks([])
            ax.set_yticks([])
            
        else:
            ax.text(0.5, 0.5, f'{person_key}\n(No Data)', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=8, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide remaining empty subplots
    for idx in range(len(display_persons), 18):
        axes[idx].set_visible(False)
    
    # Set overall title
    fig.suptitle(f'Summary: {filename}', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the summary plot
    summary_output_path = os.path.join(output_dir, 'summary_plot.png')
    plt.savefig(summary_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {summary_output_path}")
    
    plt.close()  # Close the figure to free memory

def plot_results_for_json_file(json_file_path, color_palette='tableau', target_person_id=None, plot_RP=False):
    """Plot results grouped by person (p_1, p_2, etc.) as separate plots for a single JSON file"""
    
    # Load JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract filename for title and folder creation
    filename = os.path.basename(json_file_path).replace('.json', '')
    
    # Get all evaluator names (primary keys) and organize their data
    evaluator_data = {}
    for evaluator_name in data.keys():
        if 'results' in data[evaluator_name]:
            evaluator_data[evaluator_name] = {
                'results': data[evaluator_name]['results'],
                'cam': data[evaluator_name].get('cam', 'unknown')
            }
    
    # Collect all unique person keys (p_1, p_2, etc.) across all evaluators
    all_person_keys = set()
    for evaluator_info in evaluator_data.values():
        all_person_keys.update(evaluator_info['results'].keys())


    # Sort person keys naturally
    person_keys = sorted(all_person_keys, key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else float('inf'))
    
    # Filter to specific person if requested
    if target_person_id is not None:
        if target_person_id in person_keys:
            person_keys = [target_person_id]
            print(f"Filtering to specific person: {target_person_id}")
        else:
            print(f"Warning: Person {target_person_id} not found in {filename}. Available persons: {person_keys}")
            return
    
    if not person_keys:
        print(f"No person data found in {filename}")
        return
    
    # Find global time range for consistent x-axis
    all_timestamps = []
    for evaluator_info in evaluator_data.values():
        for p_key, p_data in evaluator_info['results'].items():
            if 'timestamps' in p_data:
                all_timestamps.extend(p_data['timestamps'])
    
    if all_timestamps:
        global_start = min([convert_timestamp_to_seconds(ts) for ts in all_timestamps])
        global_end = max([convert_timestamp_to_seconds(ts) for ts in all_timestamps])
    else:
        global_start, global_end = 0, 86400  # Default to full day
    
    # Create output directory for this JSON file
    base_output_dir = 'path_to_your_analysis_root/SNH/human_evaluation_visualization'
    json_output_dir = os.path.join(base_output_dir, filename)
    os.makedirs(json_output_dir, exist_ok=True)
    
    print(f"Processing {len(person_keys)} persons for {filename} with {color_palette} palette")
    
    # Plot each person separately
    plots_created = 0
    for person_key in person_keys:
        data_plotted = plot_single_person(
            person_key, evaluator_data, global_start, global_end, 
            json_output_dir, filename, color_palette, plot_RP
        )
        if data_plotted:
            plots_created += 1
    
    print(f"Created {plots_created} individual plots in {json_output_dir}")
    
    # Create summary plot with multiple subplots
    plot_summary_figure(evaluator_data, person_keys, global_start, global_end, 
                       json_output_dir, filename, color_palette, plot_RP)
    
    print(f"Created summary plot in {json_output_dir}")


def process_single_json_file(base_path, color_palette='tableau'):
    """Process a single JSON file for debugging"""
    
    # Find all JSON files
    json_pattern = os.path.join(base_path, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No JSON files found in {base_path}")
        return
    
    # Sort files and pick the first one for debugging
    json_files.sort()
    selected_file = json_files[0]
    
    print(f"Found {len(json_files)} JSON files")
    print(f"Processing single file for debugging: {os.path.basename(selected_file)}")
    
    try:
        plot_results_for_json_file(selected_file, color_palette)
    except Exception as e:
        print(f"Error processing {selected_file}: {str(e)}")
        import traceback
        traceback.print_exc()

def process_all_json_files(base_path, color_palette='tableau'):
    """Process all JSON files in the given directory"""
    
    # Find all JSON files
    json_pattern = os.path.join(base_path, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No JSON files found in {base_path}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    # Process each file
    for json_file in sorted(json_files):
        # json_file = "path_to_your_analysis_root/SNH/0_HUMAN_RESULTS/Unify_Evaluators/human_evaluation/recording_2019_06_22_9_20_am.json"
        print(f"\nProcessing: {os.path.basename(json_file)}")
        plot_results_for_json_file(json_file, color_palette)
        # try:
        #     plot_results_for_json_file(json_file, color_palette)
        # except Exception as e:
        #     print(f"Error processing {json_file}: {str(e)}")
        # exit(0)

def show_available_palettes():
    """Display all available color palettes"""
    print("Available color palettes:")
    for name, colors in COLOR_PALETTES.items():
        print(f"  {name}: {colors}")

def main():
    # Check if user wants to see available palettes
    if SHOW_PALETTES:
        show_available_palettes()
        return
    
    # filename = 'recording_2019_07_12_6_50_am'
    # person_id = 'p_23'
    filename = None
    person_id = None
    plot_RP = False # whether to plot the recurrence plot
    
    # Path to the human evaluation data
    base_path = "path_to_your_analysis_root/SNH/0_HUMAN_RESULTS/Unify_Evaluators/human_evaluation"
    
    mode = "Debug Mode" if DEBUG_MODE else "Processing All Files"
    print(f"Starting Human Evaluation Visualization ({mode})")
    print(f"Looking for JSON files in: {base_path}")
    print(f"Using color palette: {COLOR_PALETTE}")
    
    # If specific filename and person_id are provided, process only that combination
    if filename and person_id:
        json_file_path = os.path.join(base_path, f"{filename}.json")
        if not os.path.exists(json_file_path):
            print(f"Error: JSON file not found: {json_file_path}")
            return
        print(f"Processing specific file: {filename} for person: {person_id}")
        plot_results_for_json_file(json_file_path, COLOR_PALETTE, target_person_id=person_id, plot_RP=plot_RP)
        return
    
    if DEBUG_MODE:
        # Process single file for debugging
        process_single_json_file(base_path, COLOR_PALETTE)
    else:
        # Process all files
        process_all_json_files(base_path, COLOR_PALETTE)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
