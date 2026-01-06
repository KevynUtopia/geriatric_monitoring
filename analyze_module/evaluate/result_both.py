from result_reader_evaluator2 import read_results_evaluator2
from result_reader_evaluator4 import read_results_evaluator4
from result_reader_evaluator5 import read_results_evaluator5
from result_reader_evaluator3 import read_results_evaluator3
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from evaluate.result_human_visualization import interpolate_results, hhmmss_to_seconds, seconds_to_hhmmss
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from collections import deque
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity


def format_time(x, pos):
        return seconds_to_hhmmss(int(x))


def read_human_results(base_path, file_paths):
    """
    Read and interpolate results from all human evaluators.
    
    Args:
        base_path: Base directory path for the data
        file_paths: Dictionary containing paths for each evaluator's file
    """
    # Read results from different sources
    evaluator2_results = read_results_evaluator2(f"{base_path}/{file_paths['evaluator2']}")
    evaluator4_results = read_results_evaluator4(f"{base_path}/{file_paths['evaluator4']}")
    evaluator5_results = read_results_evaluator5(f"{base_path}/{file_paths['evaluator5']}")
    evaluator3_results = read_results_evaluator3(f"{base_path}/{file_paths['evaluator3']}")

    # Interpolate results to ensure consistent time intervals
    interpolated_evaluator2_results = interpolate_results(evaluator2_results)
    interpolated_evaluator4_results = interpolate_results(evaluator4_results)
    interpolated_evaluator5_results = interpolate_results(evaluator5_results)
    interpolated_evaluator3_results = interpolate_results(evaluator3_results)
    # Combine all results and their labels
    all_results = [
        interpolated_evaluator2_results,
        interpolated_evaluator4_results,
        interpolated_evaluator5_results,
        interpolated_evaluator3_results
    ]
    labels = ['EVALUATOR2', 'EVALUATOR4', 'EVALUATOR5', 'EVALUATOR3']
    
    return all_results, labels

def read_system_results(csv_path):
    """
    Read system results from CSV file and interpolate missing timestamps.
    For gaps less than 60 seconds, interpolate values linearly.
    For gaps >= 60 seconds, fill with zeros.
    
    Args:
        csv_path: Path to the CSV file containing system results
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get the timestamp column (first column)
    time_col = df.columns[0]
    original_timestamps = df[time_col].tolist()
    
    # Convert time to seconds for interpolation
    time_seconds = df[time_col].apply(hhmmss_to_seconds)
    
    # Get measurement columns
    measurement_cols = df.columns[1:]
    
    # First standardize the data
    standardized_data = {}
    for column in measurement_cols:
        values = df[column].values
        standardized_data[column] = (values - values.min()) / (values.max() - values.min() + 1e-9)
    
    # Create interpolated data structure
    interpolated_data = {}
    interpolated_timestamps = []
    
    # Process each pair of consecutive timestamps
    for i in range(len(time_seconds) - 1):
        current_ts = time_seconds[i]
        next_ts = time_seconds[i + 1]
        current_original_ts = original_timestamps[i]
        
        # Add current timestamp
        interpolated_timestamps.append(current_original_ts)
        
        # For each measurement column
        for column in measurement_cols:
            if column not in interpolated_data:
                interpolated_data[column] = []
            interpolated_data[column].append(standardized_data[column][i])
        
        # Check if there's a gap
        gap = next_ts - current_ts
        if gap > 1:  # If there are missing seconds
            if gap < 40:  # Small gap, interpolate
                for sec in range(current_ts + 1, next_ts):
                    interpolated_ts = seconds_to_hhmmss(sec)
                    interpolated_timestamps.append(interpolated_ts)
                    
                    # Linear interpolation for each column
                    for column in measurement_cols:
                        current_val = standardized_data[column][i]
                        next_val = standardized_data[column][i + 1]
                        # Linear interpolation
                        ratio = (sec - current_ts) / gap
                        interpolated_val = current_val + ratio * (next_val - current_val)
                        interpolated_data[column].append(interpolated_val)
            else:  # Large gap, fill with zeros

                for sec in range(current_ts + 1, next_ts):
                    interpolated_ts = seconds_to_hhmmss(sec)
                    interpolated_timestamps.append(interpolated_ts)
                    
                    # Fill with zeros for each column
                    for column in measurement_cols:
                        interpolated_data[column].append(0)
    
    # Add the last timestamp and values
    interpolated_timestamps.append(original_timestamps[-1])
    for column in measurement_cols:
        interpolated_data[column].append(standardized_data[column][-1])
    
    # Finally apply smoothing to the interpolated data
    smoothed_data = {}
    for column in measurement_cols:
        values = np.array(interpolated_data[column])
        smoothed_value = gaussian_filter1d(values, sigma=13)
        smoothed_data[column] = smoothed_value
    
    return interpolated_timestamps, smoothed_data

def align_human_with_system(human_results, system_times, system_pid_key):
    """
    Align human evaluation results with system times.
    
    Args:
        human_results: List[Dict[str, Tuple[List[int], List[int]]]] - List of human evaluation results
        system_times: List[int] - System timestamps in hhmmss format
        system_pid_key: str - The person ID to align
        
    Returns:
        List[Dict[str, Tuple[List[int], List[int]]]] - Aligned human results in the same format as input
    """
    aligned_results = []
    
    # Convert system times to seconds for easier comparison
    system_times_seconds = [hhmmss_to_seconds(ts) for ts in system_times]
    system_start_sec = min(system_times_seconds)
    system_end_sec = max(system_times_seconds)
    
    for human_result in human_results:
        # Create a new dictionary for this evaluator's results
        aligned_result = {}
        
        # For each person ID in the original result
        for pid in human_result.keys():
            if pid == system_pid_key:
                # Special handling for the target person ID
                timestamps, states = human_result[pid]
                timestamps_seconds = [hhmmss_to_seconds(ts) for ts in timestamps]
                
                # Find the valid range within system times
                valid_indices = []
                for i, ts_sec in enumerate(timestamps_seconds):
                    if system_start_sec <= ts_sec <= system_end_sec:
                        valid_indices.append(i)
                
                if not valid_indices:
                    # If no valid timestamps, use system times with zeros
                    aligned_result[pid] = (system_times, [0] * len(system_times))
                    continue
                    
                # Get the valid range of timestamps and states
                valid_timestamps = [timestamps[i] for i in valid_indices]
                valid_states = [states[i] for i in valid_indices]
                
                # Create aligned states list
                aligned_states = []
                current_human_idx = 0
                
                for sys_ts in system_times:
                    sys_ts_sec = hhmmss_to_seconds(sys_ts)
                    
                    # Find the corresponding human state
                    while (current_human_idx < len(valid_timestamps) - 1 and 
                           hhmmss_to_seconds(valid_timestamps[current_human_idx + 1]) <= sys_ts_sec):
                        current_human_idx += 1
                        
                    if hhmmss_to_seconds(valid_timestamps[current_human_idx]) == sys_ts_sec:
                        # Exact match found
                        aligned_states.append(valid_states[current_human_idx])
                    else:
                        # No match found, pad with 0
                        aligned_states.append(0)
                
                aligned_result[pid] = (system_times, aligned_states)
            else:
                # For other person IDs, keep the original data
                aligned_result[pid] = human_result[pid]
        
        aligned_results.append(aligned_result)
    
    return aligned_results

def plot_comparison(system_times, system_data, aligned_human_results, human_labels, system_pid_key, save_dir):
    """
    Plot system results and human evaluation results for the same person ID.
    
    Args:
        system_times: List[int] - System timestamps
        system_data: Dict[str, List[float]] - System measurements
        aligned_human_results: List[Dict[str, Tuple[List[int], List[int]]]] - Aligned human results
        human_labels: List[str] - Labels for human evaluators
        system_pid_key: str - The person ID being compared
        save_dir: str - Directory to save the plot
    """
    # Convert system times to seconds for plotting
    system_times_seconds = [hhmmss_to_seconds(ts) for ts in system_times]
    
    # Create figure
    plt.figure(figsize=(20, 10))
    
    # Plot system data
    for key, values in system_data.items():
        plt.plot(system_times_seconds, values, label=f'System_{key}', linestyle='--', alpha=0.7)
    
    # Plot human evaluation results
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    for i, result in enumerate(aligned_human_results):
        if system_pid_key in result:
            timestamps, states = result[system_pid_key]
            timestamps_seconds = [hhmmss_to_seconds(ts) for ts in timestamps]
            plt.plot(timestamps_seconds, states, label=f'Human_{human_labels[i]}', 
                    color=colors[i], linewidth=2)
    
    # Customize plot
    plt.title(f'Comparison of System and Human Evaluation Results for {system_pid_key}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis to show time in HH:MM:SS
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    plt.xticks(rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'comparison_{system_pid_key}.jpg')
    plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_human_results(aligned_human_results, human_labels, system_times):
    """
    Analyze human evaluation results using factor analysis and CUSUM.
    
    Args:
        aligned_human_results: List[Dict[str, Tuple[List[int], List[int]]]] - Aligned human results
        human_labels: List[str] - Labels for human evaluators
        system_times: List[int] - System timestamps for alignment
        
    Returns:
        np.ndarray - Binary alarm array (1 for anomaly, 0 for normal)
    """
    # Convert aligned results to DataFrame
    df = pd.DataFrame()
    df['timestamp'] = system_times
    
    # Add each human evaluator's states as a column
    for i, result in enumerate(aligned_human_results):
        if system_pid_key in result:
            timestamps, states = result[system_pid_key]
            # Use the evaluator's label as the column name
            df[human_labels[i]] = states
    
    # Convert timestamp to datetime for indexing
    def convert_to_datetime(x):
        x = str(x).zfill(6)
        return datetime.strptime(f"2023-01-01 {x[:2]}:{x[2:4]}:{x[4:6]}", "%Y-%m-%d %H:%M:%S")
    
    df['datetime'] = df['timestamp'].apply(convert_to_datetime)
    df.set_index('datetime', inplace=True)
    
    # Get all human evaluator columns (excluding timestamp and datetime)
    human_cols = [col for col in df.columns if col in human_labels]
    print("Available human evaluator columns:", human_cols)
    
    if not human_cols:
        raise ValueError("No human evaluator data found in the DataFrame")
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[human_cols])
    
    # Check KMO and Bartlett's test
    kmo_all, kmo_model = calculate_kmo(X_scaled)
    print(f"KMO value (should be >0.6): {kmo_model}")
    
    chi2, p_value = calculate_bartlett_sphericity(X_scaled)
    print(f"Bartlett's test p-value (should be <0.05): {p_value}")
    
    # Determine number of factors
    pca = PCA()
    pca.fit(X_scaled)
    n_factors = sum(pca.explained_variance_ > 1)
    print(f"Suggested number of factors (Kaiser criterion): {n_factors}")
    
    # Perform factor analysis
    fa = FactorAnalysis(n_components=n_factors, rotation='varimax')
    fa.fit(X_scaled)
    
    # Get factor scores
    factor_scores = fa.transform(X_scaled)
    
    # Calculate composite score
    weights = pca.explained_variance_ratio_[:n_factors]
    composite_score = np.dot(factor_scores, weights)
    
    # Add composite score to DataFrame
    df['fa'] = composite_score
    
    # CUSUM parameters
    window_size = 20
    change_point = min(300, len(df['fa']))  # Use first 300 points or all if less
    mu_target = np.mean(df['fa'][:change_point])
    sigma0_sq = np.var(df['fa'][:change_point])
    k = 10  # Sensitivity parameter
    h = 50  # Threshold
    
    # Initialize CUSUM
    window = deque(maxlen=window_size)
    S_window = 0
    cusum_vals = []
    alarms = []
    
    # Run CUSUM
    for t in range(len(df['fa'])):
        # Compute squared residual
        r_t = (df['fa'][t] - mu_target) ** 2
        
        # Update window
        if len(window) == window_size:
            S_window -= (window[0] / sigma0_sq - k)
        window.append(r_t)
        S_window += (r_t / sigma0_sq - k)
        
        # Ensure non-negative
        S_window = max(0, S_window)
        cusum_vals.append(S_window)
        
        # Check threshold
        if S_window > h:
            alarms.append(t)
    
    # Generate binary alarm array
    alert_curve = np.zeros(len(df['fa']))
    for alarm in alarms:
        alert_curve[alarm] = 1

    # plot alert_curve for debugging using matplotlib
    plt.plot(alert_curve)
    plt.show()
    # exit(0)
    
    return alert_curve

def analyze_system_results(system_data_path):
    """
    Analyze system results to generate alarms using PCA, FA, and CUSUM.
    
    Args:
        system_data_path: str - Path to system data CSV file
    
    Returns:
        Tuple containing:
        - system_times: List[int] - System timestamps
        - system_alarms: np.ndarray - Binary alarm array
        - composite_score: np.ndarray - Combined score from FA
    """
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.preprocessing import StandardScaler
    from collections import deque
    
    # Read system data
    df = pd.read_csv(system_data_path)
    system_times = df['timestamp'].tolist()
    
    # Get all activity columns
    activity_columns = ['drink', 'eat', 'activity', 'sleep', 'social', 'sit', 'stand', 'watch (TV)']
    X = df[activity_columns].values
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize FA
    n_components = 3  # Number of components to keep
    fa = FactorAnalysis(n_components=n_components)
    
    # Fit FA
    fa.fit(X_scaled)
    
    # Get transformed data
    X_fa = fa.transform(X_scaled)
    
    # Calculate composite score using equal weights for FA components
    fa_weights = np.ones(n_components) / n_components  # Equal weights for FA components
    composite_score = np.dot(X_fa, fa_weights)
    
    # CUSUM parameters
    k = 10  # Reference value
    h = 20  # Decision threshold
    window_size = 20
    change_point = min(300, len(composite_score))  # Use first 300 points or all if less
    
    # Calculate target mean and variance from calibration phase
    mu_target = np.mean(composite_score[:change_point])
    sigma0_sq = np.var(composite_score[:change_point])
    
    # Initialize CUSUM
    window = deque(maxlen=window_size)
    S_window = 0
    cusum_vals = []
    alarms = []
    
    # Run CUSUM
    for t in range(len(composite_score)):
        # Compute squared residual
        r_t = (composite_score[t] - mu_target) ** 2
        
        # Update window
        if len(window) == window_size:
            S_window -= (window[0] / sigma0_sq - k)
        window.append(r_t)
        S_window += (r_t / sigma0_sq - k)
        
        # Ensure non-negative
        S_window = max(0, S_window)
        cusum_vals.append(S_window)
        
        # Check threshold
        if S_window > h:
            alarms.append(t)
    
    # Generate binary alarm array
    system_alarms = np.zeros(len(system_times))
    for alarm in alarms:
        system_alarms[alarm] = 1
    
    return system_times, system_alarms, composite_score

def plot_human_results_with_alarms(aligned_human_results, human_labels, system_times, system_alarms, composite_score, save_dir):
    """
    Plot human evaluation results and system alarm curve together.
    
    Args:
        aligned_human_results: List[Dict[str, Tuple[List[int], List[int]]]] - Aligned human results
        human_labels: List[str] - Labels for human evaluators
        system_times: List[int] - System timestamps
        system_alarms: np.ndarray - Binary alarm array from system
        composite_score: np.ndarray - Combined score from FA
        save_dir: str - Directory to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Convert system times to seconds for plotting
    system_times_seconds = [hhmmss_to_seconds(ts) for ts in system_times]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), 
                                  gridspec_kw={'height_ratios': [2, 1]})
    
    # Colors for human evaluators
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    # Plot human evaluation results in the first subplot
    for i, result in enumerate(aligned_human_results):
        if system_pid_key in result:
            timestamps, states = result[system_pid_key]
            timestamps_seconds = [hhmmss_to_seconds(ts) for ts in timestamps]
            states_array = np.array(states)  # Convert states to numpy array
            # Plot the line
            ax1.plot(timestamps_seconds, states, label=f'Human_{human_labels[i]}', 
                    color=colors[i], linewidth=2, alpha=0.7)
            # Fill the area under the curve
            ax1.fill_between(timestamps_seconds, 0, states_array, 
                           where=(states_array > 0), color=colors[i], alpha=0.2)
    
    # Customize first subplot
    ax1.set_title('Human Evaluation Results', fontsize=14, pad=20)
    ax1.set_ylabel('State', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format x-axis to show time in HH:MM:SS
    def format_time(x, pos):
        return seconds_to_hhmmss(int(x))
    
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot composite score and alarms in the second subplot
    system_times_seconds = [hhmmss_to_seconds(ts) for ts in system_times]
    
    # Plot composite score
    ax2.plot(system_times_seconds, composite_score, 
            color='blue', alpha=0.5, label='Composite Score')
    
    # Plot alarms on top
    ax2.plot(system_times_seconds, system_alarms, color='red', linewidth=2, label='System Alarms')
    
    # Fill the area under the alarm curve
    ax2.fill_between(system_times_seconds, 0, system_alarms, 
                     where=(system_alarms > 0), color='red', alpha=0.3)
    
    # Customize second subplot
    ax2.set_title('System Anomaly Detection Results', fontsize=14, pad=20)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    ax2.tick_params(axis='x', rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'human_results_with_system_alarms_{system_pid_key}.jpg')
    plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Hardcoded paths
    video_id = "recording_2019_06_27_10_30_am"
    base_path = f"evaluate/{video_id}"
    file_paths = {
        'e2': f"{video_id}_cam11_e2.txt",  # Anonymized: e2=zzr
        'e4': f"{video_id}_cam13_e4.json",  # Anonymized: e4=zww
        'e5': f"{video_id}_cam_12_e5.json",  # Anonymized: e5=roy
        'e3': f"{video_id}_cam_10_e3.json"  # Anonymized: e3=hsy
    }

    for system_pid_key in ["p_1", "p_2", "p_3", "p_4", "p_6", "p_7", "p_8", "p_9", "p_10", "p_11", "p_12", "p_13", "p_14", "p_15", "p_16", "p_17", "p_18", "p_19", "p_20", "p_21", "p_22"]:
        system_data_path = f"evaluate/alignment_soft_v5/{system_pid_key}.csv"
    
        # Read system results
        '''
            system_times: List[int]
            system_data: Dict[str, List[float]]
        '''
        system_times, system_data = read_system_results(system_data_path)
        print("System times length: ", len(system_times))
        
        # Read human evaluation results
        '''
            human_results: List[Dict[str, Tuple[List[int], List[int]]]]
                List of different evaluators, for each evaluator:
                key: pid
                value: Tuple[List[int], List[int]]
                    List[int]: timestamps
                    List[int]: states
            human_labels: List[str]
        '''
        human_results, human_labels = read_human_results(base_path, file_paths)
        print("Human evaluation results loaded successfully")
        
        # Align human results with system times
        aligned_human_results = align_human_with_system(human_results, system_times, system_pid_key)
        print("Aligned human results length: ", len(aligned_human_results))
        
        print("System results loaded successfully")
        
        # Analyze system results to get alarms
        system_times, system_alarms, composite_score = analyze_system_results(system_data_path)
        
        # Plot human results with system alarms
        save_dir = "path_to_your_analysis_root/SNH/human_evaluation/img"
        plot_human_results_with_alarms(aligned_human_results, human_labels, system_times, system_alarms, composite_score, save_dir)
        print(f"Plot saved to {save_dir}")
        
        
        
   
