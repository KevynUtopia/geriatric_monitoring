import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from collections import deque

def hhmmss_to_seconds(hhmmss):
    """Convert HHMMSS format to seconds."""
    hhmmss = str(hhmmss).zfill(6)  # Ensure 6 digits with leading zeros
    hours = int(hhmmss[:2])
    minutes = int(hhmmss[2:4])
    seconds = int(hhmmss[4:])
    return hours * 3600 + minutes * 60 + seconds

def seconds_to_hhmmss(seconds):
    """Convert seconds to HHMMSS format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}{minutes:02d}{secs:02d}"

def process_time_series(csv_path, output_path=None):
    # Read the CSV file
    df = pd.read_csv(csv_path)#[:500]
    
    # Get the timestamp column (first column)
    time_col = df.columns[0]
    
    # Convert time to seconds for plotting
    time_seconds = df[time_col].apply(hhmmss_to_seconds)
    
    # Calculate time intervals
    time_intervals = time_seconds.diff().dropna()
    avg_interval = time_intervals.mean()
    
    # Create a new DataFrame with processed values
    processed_df = pd.DataFrame()
    processed_df[time_col] = time_seconds
    
    # Process each measurement column
    for column in df.columns[1:]:
        # Step 1: Standardize the data first
        standardized_value = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        
        # Step 2: Interpolate or pad based on time intervals
        if avg_interval < 30:  # If average interval is less than 30 seconds, interpolate
            # Interpolate missing values
            interpolated_values = standardized_value.interpolate(method='linear')
            processed_df[column] = interpolated_values
        else:  # If average interval is 30 seconds or more, pad with zeros
            # Create a new series with zeros
            padded_values = standardized_value.copy()
            # Find gaps longer than 30 seconds
            gaps = time_intervals[time_intervals > 30].index
            # Pad zeros at the start of each gap
            for gap_idx in gaps:
                padded_values.iloc[gap_idx] = 0
            processed_df[column] = padded_values
    

    # Create the plot
    plt.figure(figsize=(15, 3))
    
    # Plot each measurement column
    for column in processed_df.columns[1:]:
        # Step 3: Apply Gaussian smoothing to the processed data
        smoothed_value = gaussian_filter1d(processed_df[column], sigma=7)
        plt.plot(processed_df[time_col], smoothed_value, linewidth=0.5)
        processed_df[column] = smoothed_value

    return processed_df, time_col, smoothed_value    
    
def plot_time_series(processed_df, time_col, smoothed_value, Composite_Score, alert_curve, output_path=None):
    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('Standardized and Smoothed Measurements')
    plt.title('Time Series Measurements (Standardized and Gaussian Smoothed)')
    plt.grid(True)

    plt.plot(processed_df[time_col], alert_curve, color='red', label='Alert Curve', linewidth=1)
    # fill in the area between the alert curve and x-axis
    plt.fill_between(processed_df[time_col], alert_curve, 0, color='red', alpha=0.1)

    # add a curve of Composite_Score, which is linewidth=2 while others are linewidth=0.5
    # standarize Composite_Score to 0-1
    Composite_Score = (Composite_Score - Composite_Score.min()) / (Composite_Score.max() - Composite_Score.min())
    plt.plot(processed_df[time_col], Composite_Score, color='blue', label='Composite Score', linewidth=2)
    
    # Set x-axis ticks to show HHMMSS format
    current_ticks = plt.xticks()[0]
    plt.xticks(current_ticks, [seconds_to_hhmmss(int(tick)) for tick in current_ticks], rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot if output_path is provided, otherwise show it
    if output_path:
        plt.savefig(output_path, format='jpg', dpi=500, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def factor_analysis(processed_df, time_col, smoothed_value, output_path=None):
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(processed_df)

    # Initialize FA
    n_components = 3  # Number of components to keep
    fa = FactorAnalysis(n_components=n_components)
    pca = PCA()
    # Fit FA
    fa.fit(X_scaled)
    pca.fit(X_scaled)

    factor_scores = fa.transform(X_scaled)


    weights = pca.explained_variance_ratio_[:n_components]
    Composite_Score = np.dot(factor_scores, weights)

    return Composite_Score


def cusum_analysis(Composite_Score, output_path=None):
    # CUSUM parameters
    k = 10  # Reference value
    h = 100  # Decision threshold
    window_size = 20
    change_point = min(300, len(Composite_Score))  # Use first 300 points or all if less

    # Calculate target mean and variance from calibration phase
    mu_target = np.mean(Composite_Score[:change_point])
    sigma0_sq = np.var(Composite_Score[:change_point])
    
    # Initialize CUSUM
    window = deque(maxlen=window_size)
    S_window = 0
    cusum_vals = []
    alarms = []

    # Calculate CUSUM values
    for t in range(len(Composite_Score)):
        # Compute squared residual for current point
        r_t = (Composite_Score[t] - mu_target) ** 2

        # Add to window and update cumulative sum
        if len(window) == window_size:
            # Remove oldest residual's contribution before adding new
            S_window -= (window[0] / sigma0_sq - k)
        window.append(r_t)
        S_window += (r_t / sigma0_sq - k)

        # Ensure cumulative sum doesn't drop below 0 (CUSUM reset)
        S_window = max(0, S_window)
        cusum_vals.append(S_window)

        # Check threshold
        if S_window > h:
            alarms.append(t)

    alert_curve = np.zeros(len(Composite_Score))
    for alarm in alarms:
        alert_curve[alarm] = 1
    

    return alert_curve



def main():
    parser = argparse.ArgumentParser(description='Plot time series data from a CSV file')
    parser.add_argument('--csv_path', type=str, default="evaluate/alignment_soft_v5/p_2.csv", help='Path to the CSV file')
    parser.add_argument('--output_path', type=str, help='Path to save the output plot')
    args = parser.parse_args()
    
    args.csv_path = "/Users/kevynzhang/Downloads/alignment_soft_v6/p_1.csv"
    args.output_path = "/Users/kevynzhang/Downloads/alignment_soft_v6/img/p_1_plot.png"
    
    # Generate output path if not provided
    if not args.output_path:
        args.output_path = args.csv_path.rsplit('.', 1)[0] + '_plot.png'
    
    processed_df, time_col, smoothed_value = process_time_series(args.csv_path, args.output_path)
    Composite_Score = factor_analysis(processed_df, time_col, smoothed_value, args.output_path)
    alert_curve = cusum_analysis(Composite_Score, args.output_path)

    plot_time_series(processed_df, time_col, smoothed_value, Composite_Score, alert_curve, args.output_path)
    print(f"Plot saved to: {args.output_path}")

if __name__ == '__main__':
    main()
