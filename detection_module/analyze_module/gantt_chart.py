import plotly.express as px
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, SecondLocator, MinuteLocator, date2num
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import gaussian_filter1d



path = '/Users/kevynzhang/Downloads/p_1.csv'
df = pd.read_csv(path, header=0) 

all_keys = list(df.keys())[1:]
durations = []
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
          '#59a14f', '#edc948', '#b07aa1', '#ff9da7']
colors = ['#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b',
          '#d0bbff', '#debb9b', '#fab0e4', '#cfcfcf']
colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195',
          '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52',
          '#8172b3', '#937860', '#da8bc3', '#8c8c8c']
# colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
#           '#59a14f', '#edc948', '#b07aa1', '#ff9da7']
 
# Convert timestamp to datetime (using today's date as base)
def convert_to_datetime(x):
    x = str(x).zfill(6)  # pad with leading zeros if needed
    return datetime.strptime(f"2023-01-01 {x[:2]}:{x[2:4]}:{x[4:6]}", "%Y-%m-%d %H:%M:%S")

df['datetime'] = df['timestamp'].apply(convert_to_datetime)

# Create time grid for smoothing
time_grid = pd.date_range(start=df['datetime'].min(), end=df['datetime'].max(), freq='1S')
time_numeric = date2num(time_grid)

# Function to find activity periods
def get_activity_periods(activity_series, datetime_series):
    if not isinstance(activity_series, pd.Series):
        raise ValueError("Input must be a pandas Series.")
    
    if not set(activity_series.unique()).issubset({0, 1}):
        raise ValueError("Series must contain only 0s and 1s.")
    
    # activity_series = pd.Series([1, 1, 1, 0, 0, 1, 0])
    # Initialize result with 0s
    result = pd.Series(0, index=activity_series.index)
    
    # Shift the series to compare with previous values
    prev = activity_series.shift(1)
    
    # Case 1: First element is 1 → mark 1
    if len(activity_series) > 0 and activity_series.iloc[0] == 1:
        result.iloc[0] = 1
    
    # Case 2: 0→1 transition → mark 1 (excluding the first element)
    result[(activity_series == 1) & (prev == 0)] = 1
    
    # Case 3: 1→0 transition → mark -1
    result[(activity_series == 0) & (prev == 1)] = -1

    # print(activity_series)
    # print(result)
    # exit(0)
    starts = datetime_series[result == 1]
    ends = datetime_series[result == -1]
    
    # Handle case where activity continues until end
    if len(starts) > len(ends):
        ends = pd.concat([ends, pd.Series([datetime_series.iloc[-1]])])
    
    return starts, ends



    
def smooth_activity(activity_series, datetime_series, sigma=5):
    """Apply Gaussian smoothing to activity data"""
    # Create binary signal on regular grid
    binary_signal = np.zeros_like(time_numeric)
    starts, ends = get_activity_periods(activity_series, datetime_series)
    
    for start, end in zip(starts, ends):
        mask = (time_grid >= start) & (time_grid <= end)
        binary_signal[mask] = 3.
    
    # Apply Gaussian filter
    smoothed = gaussian_filter1d(binary_signal, sigma=sigma, mode='constant')
    return smoothed



# Create chart with two rows
fig, ax = plt.subplots(figsize=(30, 4))
row_positions = [0, 1, 2, 3, 4, 5, 6, 7]
row_height = 0.8
# Get periods for both activities
for y_pos, color, key in zip(row_positions, colors, all_keys):
    
    # set background
    inactive_starts, inactive_ends = get_activity_periods(1 - df[key], df['datetime'])
    for start, end in zip(inactive_starts, inactive_ends):
        duration = (end - start).total_seconds() / (24 * 3600) 
        ax.barh(y=y_pos, width=duration, left=start, 
                height=row_height, color=color, alpha=0.05, edgecolor='none')
    
    # set each bar with smooth effect
    smoothed = smooth_activity(df[key], df['datetime'], sigma=3)
    ax.fill_between(time_grid, 
                    y_pos - row_height/2, 
                    y_pos + row_height/2, 
                    where=smoothed > 0.1,
                    color=color,
                    alpha=0.5,
                    edgecolor='none',
                    interpolate=False)
    


# After plotting all bars, add perfect row borders:

# First calculate the true data boundaries
x_min = date2num(df['datetime'].min())
x_max = date2num(df['datetime'].max())
total_width = x_max - x_min

for y_pos in row_positions:
    # Calculate exact bar edges
    bottom_edge = y_pos - row_height/2
    top_edge = y_pos + row_height/2
    
    # Create border that exactly matches the row
    rect = patches.Rectangle(
        (x_min, bottom_edge),
        total_width,
        row_height,
        linewidth=1.5,  # Slightly thicker border
        edgecolor='#333333',
        facecolor='none',
        zorder=20,  # Higher than bars
        clip_on=False  # Prevent clipping at axes edges
    )
    ax.add_patch(rect)
    
    # Add subtle horizontal line through center for better visual connection
    # ax.axhline(y=y_pos, color='#333333', linewidth=0.5, alpha=0.3, zorder=15)

# Ensure no padding is added
ax.set_xlim(x_min, x_max)
ax.autoscale(enable=False)  # Prevent auto-scaling
    
# Customize the plot
ax.set_yticks(row_positions)
ax.set_yticklabels(all_keys)
ax.set_ylim(-0.5, 8.5)  # Tight bounds around the two bars

# Configure x-axis (only at bottom)
ax.xaxis.set_major_locator(MinuteLocator(interval=30))
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
plt.xticks(rotation=45)
plt.xlabel('Time')



# Remove unnecessary spines
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)

plt.title('Combined Activity Timeline')
plt.tight_layout()
plt.show()