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
import glob

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrate_system.helper import get_color_palette
from integrate_system.gantt_chart import GanttChart
from integrate_system.factor_analysis import FactorAnalyzer
from integrate_system.model.LSTMAD import LSTMModel
import torch.nn as nn
from integrate_system.config import TimeMixerConfig_imputation, TimeMixerConfig_forecast, TimesNetConfig

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
        self.run_name = ''
        self.model_type = 'LSTM'  # Default to LSTM

        self.anomaly_criterion = nn.MSELoss()
    
    def _load_data(self):
        """Load and process the CSV data."""
        # self.df = pd.read_csv(self.path, header=0)[self.col_names]
        self.df = pd.read_csv(self.path, header=0)
        
        # Check if timestamps are already in seconds format
        sample_timestamp = self.df['timestamp'].iloc[0]
        print(f"Sample timestamp: {sample_timestamp}, type: {type(sample_timestamp)}")
        
        # Check if timestamp is already numeric (seconds format)
        if pd.api.types.is_numeric_dtype(self.df['timestamp']):
            print("Timestamps are already in numeric (seconds) format, skipping conversion")
            # Check if it's Unix timestamp (very large numbers) or seconds from start of day
            if sample_timestamp > 86400:  # More than 24 hours in seconds, likely Unix timestamp
                print("Detected Unix timestamp format")
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='s')
            else:
                print("Detected seconds from start of day format")
                # Convert seconds from start of day to datetime
                # Assume a reference date (e.g., 2019-07-01)
                base_date = pd.to_datetime('2019-07-01')
                self.df['timestamp'] = base_date + pd.to_timedelta(self.df['timestamp'], unit='s')
        else:
            print("Converting timestamps from hhmmss format to datetime")
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
    
    def factor_analysis(self, fa_base_path='', person_id=None):
        """
        Perform factor analysis on the data.
        
        Args:
            fa_base_path (str): Base path where FA models are stored
            person_id (str): Person identifier (e.g., 'p_1', 'p_2')
        """
        # Extract person ID from file path if not provided
        if person_id is None:
            person_id = self._extract_person_id_from_path()
        
        analyzer = FactorAnalyzer(self.df, self.all_keys, person_id=person_id, model_base_path=fa_base_path)
        alert_curve, composite_score, variance_curve = analyzer.perform_analysis()
        return alert_curve, composite_score, variance_curve
    
    def _extract_person_id_from_path(self):
        """
        Extract person ID from the current file path.
        Supports multiple filename formats.
        
        Returns:
            str: Person ID (e.g., 'p_1') or None if not found
        """
        return self._extract_person_id_from_filename(self.path)
    
    def initialize_anomaly_detection(self, model_type='LSTM', window_size=100, input_size=46, hidden_dim=64, pred_len=10, num_layers=2, batch_size=16, num_epochs=100, learning_rate=0.001):
        """
        Initialize anomaly detection model.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        if model_type == 'LSTM':
            # Create LSTM model
            self.model = LSTMModel(window_size=window_size, input_size=input_size,
                                   hidden_dim=hidden_dim, pred_len=pred_len, num_layers=num_layers, 
                                   batch_size=batch_size, device=device).to(device)
            
            # Load LSTM weights
            path = 'integrate_system/model/checkpoint_step_690000.pth'
            weights = torch.load(path, map_location=torch.device('cpu'))['model_state_dict']
            self.model.load_state_dict(weights)
            
        elif model_type == 'TimeMixer':
            # Imputation model (multivariate)
            from integrate_system.models.TimeMixer import Model as TimeMixerModel
            config_impute = TimeMixerConfig_imputation()
            config_impute.enc_in = 46
            config_impute.c_out = 46
            self.model_timemixer_impute = TimeMixerModel(config_impute).to(device)
            impute_ckpt = 'integrate_system/model/imputation/checkpoint2.pth'
            if os.path.exists(impute_ckpt):
                checkpoint = torch.load(impute_ckpt, map_location=device)
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
                self.model_timemixer_impute.load_state_dict(state_dict, strict=False)
                print("TimeMixer imputation model loaded successfully")
            else:
                print(f"Warning: Imputation checkpoint not found at {impute_ckpt}")
            
            
            # Forecasting model (univariate)
            config_forecast = TimeMixerConfig_forecast()

            self.model_timemixer_forecast = TimeMixerModel(config_forecast).to(device)
            forecast_ckpt = 'integrate_system/model/forecast/checkpoint.pth'
            if os.path.exists(forecast_ckpt):
                checkpoint = torch.load(forecast_ckpt, map_location=device)
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
                self.model_timemixer_forecast.load_state_dict(state_dict, strict=False)
                print("TimeMixer forecasting model loaded successfully")
            else:
                print(f"Warning: Forecasting checkpoint not found at {forecast_ckpt}")
        
        elif model_type == 'TimesNet':
            from integrate_system.models.TimesNet import Model as TimesNetModel
            config_timesnet = TimesNetConfig()
            self.model_timesnet = TimesNetModel(config_timesnet).to(device)
            timesnet_ckpt = 'integrate_system/model/timesnet/checkpoint.pth'
            if os.path.exists(timesnet_ckpt):
                checkpoint = torch.load(timesnet_ckpt, map_location=device)
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
                self.model_timesnet.load_state_dict(state_dict, strict=False)
                print("TimesNet model loaded successfully")
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'LSTM' or 'TimeMixer'.")

    def anomaly_detection(self, data, threshold=0.05):
        """
        Perform anomaly detection on the data using sliding windows.
        
        Args:
            data (torch.Tensor): Input data tensor of shape (n_samples, n_features)
            threshold (float): Threshold for anomaly detection
            
        Returns:
            numpy.ndarray: Anomaly scores for each time step
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        
        if self.model_type == 'LSTM':
            return self._lstm_anomaly_detection(data, threshold)
        elif self.model_type == 'TimeMixer':
            return self._timemixer_anomaly_detection(data, threshold)
        elif self.model_type == 'TimesNet':
            return self._timesnet_anomaly_detection(data, threshold)
    
    def _lstm_anomaly_detection(self, data, threshold=0.05):
        """LSTM-based anomaly detection."""
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
        
        # Convert scores to 0 or 1 according to the threshold
        ad_results = (padded_scores > threshold).astype(int)
        return ad_results, padded_scores
    
    def _timemixer_anomaly_detection(self, data, threshold=0.05):
        """TimeMixer-based anomaly detection using both imputation and forecasting."""
        # Imputation config
        window_size = 30  # Imputation model seq_len
        # Forecasting config
        forecast_seq_len = 200
        label_len = 48
        pred_len = 96
        stride = 1
        n_samples, n_features = data.shape
        scores = []
        self.model_timemixer_impute.eval()
        self.model_timemixer_forecast.eval()
        device = data.device
        with torch.no_grad():
            for i in range(0, n_samples - max(window_size, forecast_seq_len) - pred_len + 1, stride):
                # --- Imputation ---
                if i + window_size > n_samples:
                    break
                input_window = data[i:i + window_size]  # (30, n_features)
                input_batch = input_window.unsqueeze(0)  # (1, 30, n_features)
                B, T, N = input_batch.shape
                mask = torch.rand((B, T, N), device=input_batch.device)
                mask_rate = 0.01
                mask = (mask > mask_rate).float()
                masked_input = input_batch.masked_fill(mask == 0, 0)
                preds_impute = self.model_timemixer_impute.imputation(masked_input, None, mask)
                mse_impute = torch.mean((preds_impute - input_batch).pow(2))
                
                # --- Forecasting ---
                # if i + forecast_seq_len + pred_len > n_samples:
                #     break
                # # Encoder input: first forecast_seq_len data
                # x_enc = data[i:i + forecast_seq_len].unsqueeze(0)  # (1, 200, 46)                
                # # Decoder input: [last_label_len_data, zeros] (matching training)
                # batch_y = data[i + forecast_seq_len:i + forecast_seq_len + label_len].unsqueeze(0)
                # dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :pred_len, :], dec_inp], dim=1).float()
                # # Get model prediction
                # preds_forecast = self.model_timemixer_forecast.forecast(x_enc, None, dec_inp, None)

                # # Target: actual future data
                # target = data[i + forecast_seq_len:i + forecast_seq_len + pred_len]  # (96, 46)
                
                # # Compare prediction with target
                # mse_forecast = torch.mean((preds_forecast.squeeze(0) - target).pow(2))
                mse_forecast = 0.0

                # --- Combine ---
                mse_sum = mse_impute + mse_forecast
                scores.append(mse_sum.item())
        # Pad scores to match original data length
        scores = np.array(scores)
        padded_scores = np.zeros(n_samples)
        pad_start = max(window_size, forecast_seq_len) + pred_len
        
        # Ensure we don't exceed the bounds
        end_idx = min(pad_start + len(scores), n_samples)
        actual_scores_length = end_idx - pad_start
        
        if actual_scores_length > 0:
            padded_scores[pad_start:end_idx] = scores[:actual_scores_length]
        
        ad_results = (padded_scores > threshold).astype(int)
        return ad_results, padded_scores
    
    def _timesnet_anomaly_detection(self, data, threshold=0.05):
        """TimesNet-based anomaly detection."""
        # Imputation config
        window_size = 200  # Imputation model seq_len
        stride = 1
        n_samples, n_features = data.shape
        scores = []
        self.model_timesnet.eval()

        device = data.device
        attens_energy = []

        with torch.no_grad():
            for i in range(0, n_samples - window_size, stride):

                if i + window_size > n_samples:
                    break
                input_window = data[i:i + window_size]  # (30, n_features)
                input_batch = input_window.unsqueeze(0)  # (1, 30, n_features)
                B, T, N = input_batch.shape

                input_batch = input_batch.float().to(device)
                # reconstruction
                outputs = self.model_timesnet(input_batch, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(input_batch, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
        
        # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        padded_scores = np.zeros(n_samples)
        pad_start = window_size
        
        # Ensure we don't exceed the bounds
        end_idx = min(pad_start + len(test_energy), n_samples)
        actual_scores_length = end_idx - pad_start
        
        if actual_scores_length > 0:
            padded_scores[pad_start:end_idx] = test_energy[:actual_scores_length]
        
        ad_results = (padded_scores > threshold).astype(int)

        return ad_results, padded_scores

    def process_directory(self, directory_path, debug_mode=False, min_csv_length=500, output_dir=None, model_type='LSTM'):
        """
        Process all CSV files in the specified directory.
        
        Args:
            directory_path (str): Path to the directory containing CSV files
            debug_mode (bool): If True, process only one file for debugging
            min_csv_length (int): Minimum number of rows required in CSV
            output_dir (str): Output directory for results
            model_type (str): Type of model to use ('LSTM' or 'TimeMixer')
            
        Returns:
            dict: Dictionary with processing results for each file
        """
        if not os.path.exists(directory_path):
            print(f"Error: Directory does not exist: {directory_path}")
            return {}
        
        # Find all CSV files in the directory
        csv_pattern = os.path.join(directory_path, "*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"No CSV files found in {directory_path}")
            return {}
        
        print(f"Found {len(csv_files)} CSV files in {directory_path}")
        print(f"Using model type: {model_type}")
        
        results = {}
        processed_count = 0
        skipped_count = 0
        fa_model_base_path = self.run_name if self.run_name else 'trained_fa_models'
        
        for csv_file in sorted(csv_files):
            # Debug mode: specify a specific csv file
            if debug_mode:
                csv_file = 'path_to_your_analysis_root/SNH/TSAD_DATASET/ts_recording_2019_06_22_9_20_am_p_5.csv'
            
            # Skip files with "giver" in the name
            if "giver" in csv_file:
                print(f"Skipping giver file: {os.path.basename(csv_file)}")
                skipped_count += 1
                continue
            
            filename = os.path.basename(csv_file).replace('.csv', '')
            
            # Extract person ID BEFORE processing
            person_id = self._extract_person_id_from_filename(csv_file)
            if person_id is None:
                print(f"WARNING: Could not extract person ID from {filename}, skipping file")
                skipped_count += 1
                continue
            
            # Check if FA model exists for this person
            fa_model_path = os.path.join(fa_model_base_path, f'fa_model_{person_id}.joblib')
            if not os.path.exists(fa_model_path):
                print(f"WARNING: FA model not found for {person_id} at {fa_model_path}, skipping file")
                skipped_count += 1
                continue
            
            print(f"Processing CSV: {filename} (Person ID: {person_id})")
            
            # Check CSV length first
            df_check = pd.read_csv(csv_file)
            if len(df_check) < min_csv_length:
                print(f"Skipping {filename}: Only {len(df_check)} rows (minimum {min_csv_length} required)")
                skipped_count += 1
                continue
            
            print(f"CSV has {len(df_check)} rows, proceeding with analysis...")
            print(f"Using FA model: {fa_model_path}")
            
            # Process the file with specified model type
            file_results = self.process_data(csv_file, model_type=model_type)
            self.save_results(file_results, filename, output_dir=output_dir)
            
            results[filename] = {
                'person_id': person_id,
                'file_path': csv_file,
                'fa_model_path': fa_model_path,
                'model_type': model_type,
                'status': 'success'
            }
            
            processed_count += 1
            
            # Debug mode: only process one file
            if debug_mode:
                print("Debug mode: stopping after first file")
                break
        
        print(f"\n{'='*80}")
        print(f"SUMMARY:")
        print(f"Total CSV files found: {len(csv_files)}")
        print(f"Total CSV files processed: {processed_count}")
        print(f"Total CSV files skipped: {skipped_count}")
        print(f"Model type used: {model_type}")
        print(f"{'='*80}")
        
        return results

    def _extract_person_id_from_filename(self, file_path):
        """
        Extract person ID from a given file path.
        Supports multiple filename formats.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            str: Person ID (e.g., 'p_1') or None if not found
        """
        if not file_path:
            return None
            
        filename = os.path.basename(file_path)
        
        # Remove .csv extension for pattern matching
        basename = filename.replace('.csv', '')
        
        import re
        
        # Pattern 1: ts_recording_YYYY_MM_DD_H_MM_XM_p_N.csv
        pattern1 = r'ts_recording_\d{4}_\d{2}_\d{2}_\d+_\d+_[ap]m_p_(\d+)'
        match1 = re.search(pattern1, basename)
        if match1:
            person_id = f"p_{match1.group(1)}"
            return person_id
        
        # Pattern 2: Any filename containing p_N
        pattern2 = r'p_(\d+)'
        match2 = re.search(pattern2, basename)
        if match2:
            person_id = f"p_{match2.group(1)}"
            return person_id
            
        return None

    def process_data(self, path, model_type='LSTM'):
        """
        Process the data for anomaly detection.
        
        Args:
            path (str): Path to the CSV file containing the data
            model_type (str): Type of model to use ('LSTM' or 'TimeMixer')
            
        Returns:
            dict: Dictionary containing processing results including combined and individual components
        """
        self.path = path
        self._load_data()
        # self.smooth_data()
        
        # Initialize anomaly detection with specified model type
        self.initialize_anomaly_detection(model_type=model_type)
        
        # Convert data to tensor (exclude timestamp column if it exists)
        if 'timestamp' in self.df.columns:
            # If smooth_data was not called, exclude timestamp column
            data = torch.tensor(self.df[self.all_keys].values, dtype=torch.float32)
        else:
            # If smooth_data was called, timestamp is now index, use all values
            data = torch.tensor(self.df.values, dtype=torch.float32)
        
        # Perform anomaly detection
        ad_results, anomaly_scores = self.anomaly_detection(data)
        
        # Perform factor analysis with trained models
        # Use trained_fa_models directory if run_name is not specified or use run_name as base path
        fa_base_path = self.run_name if self.run_name else 'fa_weights'
        
        # Extract person ID from the file path
        person_id = self._extract_person_id_from_path()
        if person_id:
            print(f"Processing Factor Analysis for {person_id} using models from: {fa_base_path}")
        else:
            print(f"Warning: Could not extract person ID from {path}, FA may not work correctly")
        
        alert_curve, composite_score, variance_curve = self.factor_analysis(fa_base_path=fa_base_path, person_id=person_id)
        
        # Combine results
        combined_results, anomaly_scores, variance_curve = self.combine_results(anomaly_scores, alert_curve, variance_curve, composite_score, ratio = [0.5, 0., 0.5])
        
        # Return both combined and individual components
        return {
            'combined': combined_results,
            'anomaly_scores': anomaly_scores,
            'alert_curve': alert_curve,
            'variance_curve': variance_curve,
            'composite_score': composite_score,
            'model_type': self.model_type
        }

    def combine_results(self, anomaly_scores, alert_curve, variance_curve, composite_score, ratio = [0.5, 0.3, 0.2]):
        """
        Combine the results of the anomaly detection and factor analysis.
        
        Args:
            anomaly_scores (numpy.ndarray): Anomaly detection scores
            alert_curve (numpy.ndarray): Alert curve from factor analysis
            variance_curve (numpy.ndarray): Variance curve from factor analysis
            composite_score (numpy.ndarray): Composite score from factor analysis
            ratio (list): Weighting ratios for combining signals
            
        Returns:
            tuple: Combined alarm signal, normalized anomaly scores, normalized variance curve
        """
        # Combine the signals using weighted sum
        # alarm = ratio[0] * anomaly_scores + ratio[1] * alert_curve + ratio[2] * variance_curve
        # normalize anomaly_scores and variance_curve to 0-1
        if anomaly_scores.max() - anomaly_scores.min() == 0:
            anomaly_scores = np.zeros_like(anomaly_scores)
        else:
            anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        if variance_curve.max() - variance_curve.min() == 0:
            variance_curve = np.zeros_like(variance_curve)
        else:
            variance_curve = (variance_curve - variance_curve.min()) / (variance_curve.max() - variance_curve.min())
        # max of anomaly_scores and variance_curve
        # alarm = np.maximum(anomaly_scores, variance_curve)
        # alarm = np.minimum(anomaly_scores, variance_curve)
        alarm = anomaly_scores
        # normalize alarm to 0-1
        if alarm.max() - alarm.min() == 0:
            alarm = np.zeros_like(alarm)
        else:
            alarm = (alarm - alarm.min()) / (alarm.max() - alarm.min())
        return alarm, anomaly_scores, variance_curve

    def save_results(self, file_results, filename, output_dir = None):
        """
        Save the results as a plot and csv files.
        Saves both combined results and separate components.
        """
        if output_dir is None:
            output_dir = 'path_to_your_analysis_root/SNH/alarm_result_output_v5'
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Extract combined results and individual components
        combined_alarm = file_results['combined']
        anomaly_scores = file_results['anomaly_scores']
        alert_curve = file_results['alert_curve']
        variance_curve = file_results['variance_curve']
        composite_score = file_results['composite_score']
        model_type = file_results.get('model_type', 'Unknown')
        
        # Handle both cases: timestamps in index (after smooth_data) or in column (no smooth_data)
        if 'timestamp' in self.df.columns:
            # Timestamps are in column (smooth_data was not called)
            timestamps = self.df['timestamp']
            x_axis_data = range(len(self.df))  # Use row indices for plotting
        else:
            # Timestamps are in index (smooth_data was called)
            timestamps = self.df.index
            x_axis_data = self.df.index
        
        plt.figure(figsize=(10, 4))
        plt.plot(x_axis_data, combined_alarm)
        # x-axis is the timestamp
        plt.xlabel('Timestamp')
        plt.ylabel('Alarm')
        plt.title(f'Anomaly Detection Results - {model_type} Model')
        
        # Convert timestamps to seconds for smart tick placement
        def timestamp_to_seconds(ts_str):
            time_part = ts_str.split(' ')[1]  # Get "HH:MM:SS"
            h, m, s = map(int, time_part.split(':'))
            return h * 3600 + m * 60 + s
        
        def seconds_to_timestamp(seconds):
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
        # Get time range in seconds
        time_seconds = [timestamp_to_seconds(ts) for ts in timestamps]
        global_start = min(time_seconds)
        global_end = max(time_seconds)
        time_range = global_end - global_start
        
        # Determine tick interval based on time range (similar to result_human_visualization.py)
        if time_range > 7200:  # More than 2 hours
            tick_interval = 3600  # Every hour
        elif time_range > 3600:  # More than 1 hour
            tick_interval = 1800  # Every 30 minutes
        elif time_range > 1800:  # More than 30 minutes
            tick_interval = 900   # Every 15 minutes
        elif time_range > 900:   # More than 15 minutes
            tick_interval = 300   # Every 5 minutes
        else:
            tick_interval = 60    # Every minute
        
        # Generate tick positions
        start_tick = global_start - (global_start % tick_interval)
        if start_tick < global_start:
            start_tick += tick_interval
        
        tick_positions = list(range(start_tick, global_end + tick_interval, tick_interval))
        tick_positions = [t for t in tick_positions if global_start <= t <= global_end]
        
        # Create labels in HH:MM:SS format
        tick_labels = [seconds_to_timestamp(t) for t in tick_positions]
        
        # Map tick positions back to x-axis values (using index positions)
        tick_x_positions = []
        for tick_sec in tick_positions:
            # Find the closest timestamp in our data
            closest_idx = min(range(len(time_seconds)), 
                             key=lambda i: abs(time_seconds[i] - tick_sec))
            if 'timestamp' in self.df.columns:
                # Use row index for x-axis positioning
                tick_x_positions.append(closest_idx)
            else:
                # Use dataframe index
                tick_x_positions.append(self.df.index[closest_idx])
        
        plt.xticks(tick_x_positions, tick_labels, rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename}_{model_type.lower()}.png'), dpi=300)
        plt.close()

        # Convert timestamps to seconds from start of day
        seconds = []
        for timestamp in timestamps:
            # Extract time part from format like "07-01 HH:MM:SS"
            time_part = timestamp.split(' ')[1]  # Get "HH:MM:SS"
            h, m, s = map(int, time_part.split(':'))
            total_seconds = h * 3600 + m * 60 + s
            seconds.append(total_seconds)
        
        # Save combined results (original functionality)
        df_combined = pd.DataFrame({
            'timestamp': timestamps, 
            'seconds': seconds,
            'alarm': combined_alarm,
            'model_type': model_type
        })
        df_combined.to_csv(os.path.join(output_dir, f'{filename}_{model_type.lower()}.csv'), index=False)
        
        # Save separate components as a new CSV file
        df_separate = pd.DataFrame({
            'timestamp': timestamps,
            'seconds': seconds,
            'anomaly_scores_0': anomaly_scores,
            'alert_curve': alert_curve,
            'variance_curve_1': variance_curve,
            'composite_score': composite_score,
            'model_type': model_type
        })
        df_separate.to_csv(os.path.join(output_dir, f'{filename}_{model_type.lower()}_separate.csv'), index=False)

# Example usage:
if __name__ == "__main__":
    # Process a single file with LSTM model
    # single_file_path = 'path_to_your_analysis_root/SNH/TSAD_DATASET/ts_recording_2019_06_22_9_20_am_p_5.csv'
    # engine = SystemEngine()
    # results = engine.process_data(single_file_path, model_type='LSTM')
    
    # Process a single file with TimeMixer model
    # single_file_path = 'path_to_your_analysis_root/SNH/TSAD_DATASET/ts_recording_2019_06_22_9_20_am_p_5.csv'
    # engine = SystemEngine()
    # results = engine.process_data(single_file_path, model_type='TimeMixer')
    
    # Process all CSV files in a directory
    # directory_path = 'path_to_your_analysis_root/SNH/TSAD_DATASET'
    directory_path = 'path_to_your_analysis_root/SNH/0_SYSTEM_RESULTS/detection_output/results_v7_alignment_soft/DATASET'
    output_dir = 'path_to_your_analysis_root/SNH/0_SYSTEM_RESULTS/alarm_result_output/alarm_result_output_v13_anm'
    engine = SystemEngine()
    # Set the path to the trained FA models directory
    engine.run_name = 'fa_weights2'
    
    # Process with LSTM model
    # all_results_lstm = engine.process_directory(directory_path, debug_mode=False, min_csv_length=500, output_dir=output_dir, model_type='LSTM')
    
    # Process with TimeMixer model (will fallback to LSTM if dependencies are missing)
    all_results_timemixer = engine.process_directory(directory_path, debug_mode=False, min_csv_length=500, output_dir=output_dir, model_type='TimesNet')
    
    # print(f"Processed {len(all_results_lstm)} files with LSTM model successfully")
    print(f"Processed {len(all_results_timemixer)} files with TimeMixer model successfully")
