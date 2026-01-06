import pandas as pd
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, MinuteLocator
from matplotlib.dates import DateFormatter, MinuteLocator, date2num
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import get_color_palette
from gantt_chart import GanttChart
from factor_analysis import FactorAnalyzer
from LSTMADalpha.LSTMADalpha import LSTMModel
import torch.nn as nn

class SystemEngine:
    def __init__(self):
        """
        Initialize the SystemEngine with the path to the CSV file.
        
        Args:
            path (str): Path to the CSV file containing the data
        """
        self.path = ''
        self.col_names = ['timestamp', 'drink', 'eat', 'activity', 'sleep', 'social', 'sit', 'stand', 'watch (TV)']
        self.df = None
        self.all_keys = None
    
    def _load_data(self):
        """Load and process the CSV data."""
        # self.df = pd.read_csv(self.path, header=0)[self.col_names]
        self.df = pd.read_csv(self.path, header=0)
        # Convert timestamp to datetime
        self.df['timestamp'] = self.df['timestamp'].apply(self._parse_timestamp)
        # Format timestamp to remove year
        self.df['timestamp'] = self.df['timestamp'].dt.strftime('%m-%d %H:%M:%S')
        self.all_keys = list(self.df.keys())[1:]
    
    def _parse_timestamp(self, x):
        """
        Parse timestamp from hhmmss format to datetime.
        
        Args:
            x (str): Timestamp in hhmmss format
            
        Returns:
            datetime: Parsed datetime object
        """
        x = str(x).zfill(6)  # pad with leading zeros if needed
        
        # Extract hours, minutes, seconds
        hours = int(x[:2])
        minutes = int(x[2:4])
        seconds = int(x[4:6])
        
        # Validate and fix invalid timestamps
        if seconds >= 60:
            # Convert excess seconds to minutes
            extra_minutes = seconds // 60
            seconds = seconds % 60
            minutes += extra_minutes
        
        if minutes >= 60:
            # Convert excess minutes to hours
            extra_hours = minutes // 60
            minutes = minutes % 60
            hours += extra_hours
        
        # Ensure hours don't exceed 23
        hours = hours % 24
        
        try:
            return datetime.strptime(f"07-01 {hours:02d}:{minutes:02d}:{seconds:02d}", "%m-%d %H:%M:%S")
        except ValueError as e:
            # If still invalid, return a default timestamp
            print(f"Warning: Invalid timestamp {x} (original), using default timestamp")
            return datetime.strptime("07-01 00:00:00", "%m-%d %H:%M:%S")
    
    def get_dataframe(self):
        """Return the processed dataframe."""
        return self.df
    
    def get_keys(self):
        """Return the list of activity keys."""
        return self.all_keys
    
    def get_color_space(self, palette_name='default'):
        """
        Get color space for visualization.
        
        Args:
            palette_name (str): Name of the color palette ('default', 'pastel', or 'vibrant')
            
        Returns:
            list: List of color hex codes
        """
        return get_color_palette(palette_name)
    
    def smooth_data(self, sigma=3):
        """
        Smooth the data using a Gaussian filter.
        
        Args:
            sigma (float): Standard deviation for Gaussian filter
        """
        frames = {'date': self.df['timestamp']}
        for key in self.all_keys:
            if key == 'activity':
                self.df[key] = 1.*(self.df[key]>0)
            col = gaussian_filter1d(self.df[key], sigma=sigma)
            frames[key] = col

        df = pd.DataFrame(frames)
        df.set_index('date', inplace=True)
        self.df = df

    def _create_colormap(self, color):
        """
        Create a colormap from a single color.
        
        Args:
            color (str): Hex color code
            
        Returns:
            matplotlib.colors.LinearSegmentedColormap: Colormap object
        """
        return LinearSegmentedColormap.from_list('custom', ['white', color])

    def get_gantt_chart(self, palette_name='default'):
        """
        Create and return a gantt chart figure.

        Args:
            palette_name (str): Name of the color palette to use

        Returns:
            matplotlib.figure.Figure: The gantt chart figure
        """
        colors = self.get_color_space(palette_name)
        gantt = GanttChart(self.df, self.all_keys, colors)
        return gantt.create_chart()
    
    def factor_analysis(self):
        """
        Perform factor analysis on the data.
        """
        analyzer = FactorAnalyzer(self.df, self.all_keys)
        alert_curve, composite_score = analyzer.perform_analysis()
        return alert_curve, composite_score
    

    def initialize_anomaly_detection(self, window_size=100, input_size=46, hidden_dim=64, pred_len=10, num_layers=2, batch_size=16, num_epochs=100, learning_rate=0.001):
        """
        Perform deep learning anomaly detection on the data.
        """
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        self.model  = LSTMModel(window_size=window_size,input_size=input_size,
                                hidden_dim=hidden_dim, pred_len=pred_len, num_layers=num_layers, batch_size=batch_size, device=device).to(device)

        # load weights according to the path
        path = ('path_to_your_root/results/analyze_soft/model_outputs/'
                'lstm_run_20250618_021736/checkpoint_step_19610000.pth')
        weights = torch.load(path)['model_state_dict']

        self.model.load_state_dict(weights)


    def anomaly_detection(self, data, threshold=0.05):
        """
        Perform anomaly detection on the data using sliding windows.
        
        Args:
            data (torch.Tensor): Input data tensor of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Anomaly scores for each time step
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        window_size = 100
        pred_len = 10
        stride = 1
        
        n_samples, n_features = data.shape
        scores = []
        
        self.model.eval()
        with torch.no_grad():
            # Sliding window approach
            for i in range(0, n_samples - window_size - pred_len + 1, stride):
                # Extract input window (100 points)
                input_window = data[i:i + window_size]  # Shape: (100, n_features)
                
                # Extract target (next 10 points)
                target = data[i + window_size:i + window_size + pred_len]  # Shape: (10, n_features)
                
                # Add batch dimension
                input_batch = input_window.unsqueeze(0)  # Shape: (1, 100, n_features)
                target_batch = target.unsqueeze(0)  # Shape: (1, 10, n_features)
                
                # Make prediction
                preds = self.model(input_batch)  # Shape: (1, 10, n_features)
                # Calculate mean squared error for this window
                mse = torch.mean((preds - target_batch).pow(2))
                scores.append(mse.cpu().item())
        
        # Convert to numpy array and pad to match original data length
        scores = np.array(scores)
        
        # Pad scores to match original data length
        # For the first window_size points, we don't have predictions, so use zeros
        padded_scores = np.zeros(n_samples)
        padded_scores[window_size:window_size + len(scores)] = scores
        # DEBUG: plot padded_scores as a line plot
        # convert scores to 0 or 1 according to the threshold
        ad_results = (padded_scores > threshold).astype(int)
        return ad_results, padded_scores

    def process_data(self, path):
        """
        Process the data for anomaly detection.
        
        Args:
            path (str): Path to the CSV file containing the data
        """
        self.path = path
        self._load_data()
        self.smooth_data()
        
        # Initialize anomaly detection
        self.initialize_anomaly_detection()
        # Convert data to tensor
        data = torch.tensor(self.df.values, dtype=torch.float32)
        # Perform anomaly detection
        ad = self.anomaly_detection(data)



        # ad = np.zeros(data.shape[0])
        # alert_curve, composite_score = self.factor_analysis()
        # results = self.combine_results(ad, alert_curve, composite_score)
        
        # return results
    
    def combine_results(self, ad, alert_curve, composite_score):
        """
        Combine the results of the anomaly detection and factor analysis.
        
        Args:
            ad (torch.Tensor): Anomaly detection results
            alert_curve (torch.Tensor): Alert curve from factor analysis
            composite_score (torch.Tensor): Composite score from factor analysis
            
        Returns:
            torch.Tensor: Combined alarm signal
        """
        # Combine the signals using weighted sum
        alarm = 0.5 * ad + 0.3 * alert_curve + 0.2 * composite_score
        return alarm

# Example usage:
if __name__ == "__main__":
    path = '/Users/kevynzhang/科研工作/SNH/TSAD_DATASET'
    engine = SystemEngine()
    engine.process_data(path)
