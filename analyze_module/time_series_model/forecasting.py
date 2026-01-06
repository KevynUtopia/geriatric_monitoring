import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from model import TimeSeriesForecastingModel


class TimeSeriesDataLoader:
    """Simplified data loader with minimal logging."""
    
    def __init__(self, data_directory: str, splits_file: str, min_csv_length: int = 300):
        self.data_directory = data_directory
        self.splits_file = splits_file
        self.min_csv_length = min_csv_length
        self.splits_data = self._load_splits()
        self._all_split_files = None
    
    def _load_splits(self) -> Dict:
        """Load train/val/test splits."""
        with open(self.splits_file, 'r') as f:
            return json.load(f)
    
    def _parse_timestamp(self, x) -> str:
        """Parse timestamp format."""
        x = str(x).zfill(6)
        hours = int(x[:2]) % 24
        minutes = int(x[2:4]) % 60
        seconds = int(x[4:6]) % 60
        return f"2019-07-01 {hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _load_and_process_csv(self, csv_path: str) -> Optional[pd.DataFrame]:
        """Load and process a single CSV file."""
        df = pd.read_csv(csv_path, header=0)
        
        if len(df) < self.min_csv_length:
            return None
        
        # Parse timestamps
        if pd.api.types.is_numeric_dtype(df['timestamp']):
            if df['timestamp'].iloc[0] > 86400:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                base_date = pd.to_datetime('2019-07-01')
                df['timestamp'] = base_date + pd.to_timedelta(df['timestamp'], unit='s')
        else:
            df['timestamp'] = df['timestamp'].apply(self._parse_timestamp)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Process activity columns
        activity_columns = [col for col in df.columns if col != 'timestamp']
        for col in activity_columns:
            if col == 'activity':
                df[col] = 1.0 * (df[col] > 0)
            df[col] = gaussian_filter1d(df[col], sigma=3)
        
        return df
    
    def _reduce_to_1d(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce multi-dimensional data to 1D using Factor Analysis."""
        feature_cols = [c for c in df.columns if c != 'timestamp']
        X = df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        fa = FactorAnalysis(n_components=1, random_state=42)
        fa1 = fa.fit_transform(X_scaled).reshape(-1)
        
        return pd.DataFrame({
            'timestamp': df['timestamp'],
            'fa1': fa1
        })
    
    def _collect_all_split_files(self) -> Dict[str, List[Dict]]:
        """Collect file entries for each split."""
        persons_data = self.splits_data.get('persons', {})
        all_train_files = []
        all_val_files = []
        all_test_files = []
        
        for person_id, splits in persons_data.items():
            for entry in splits.get('train', {}).get('files', []):
                entry = dict(entry)
                entry['person_id'] = person_id
                all_train_files.append(entry)
            for entry in splits.get('val', {}).get('files', []):
                entry = dict(entry)
                entry['person_id'] = person_id
                all_val_files.append(entry)
            for entry in splits.get('test', {}).get('files', []):
                entry = dict(entry)
                entry['person_id'] = person_id
                all_test_files.append(entry)
        
        self._all_split_files = {
            'train': all_train_files,
            'val': all_val_files,
            'test': all_test_files,
        }
        return self._all_split_files
    
    def iter_split_dfs(self, split_name: str):
        """Iterate over processed DataFrames for a given split."""
        if self._all_split_files is None:
            self._collect_all_split_files()
        
        files = self._all_split_files.get(split_name, [])
        for file_entry in files:
            filepath = os.path.join(self.data_directory, os.path.basename(file_entry['path']))
            df = self._load_and_process_csv(filepath)
            if df is not None:
                yield file_entry.get('person_id', 'unknown'), file_entry['filename'], df
    
    def reduce_all_files(self, output_dir: str):
        """Reduce all files to 1D and save to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name in ['train', 'val', 'test']:
            for person_id, filename, df in self.iter_split_dfs(split_name):
                reduced_df = self._reduce_to_1d(df)
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_fa1.csv")
                reduced_df.to_csv(output_path, index=False)
    
    def load_reduced_file(self, filepath: str) -> pd.DataFrame:
        """Load a reduced 1D file."""
        df = pd.read_csv(filepath)
        if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df


class TimeSeriesForecastingPipeline:
    """Simplified forecasting pipeline."""
    
    def __init__(self, data_dir: str, model_dir: str, splits_file: str, reduced_dir: str):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.splits_file = splits_file
        self.reduced_dir = reduced_dir
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(reduced_dir, exist_ok=True)
        
        self.loader = TimeSeriesDataLoader(data_dir, splits_file)
        self.model = None
    
    def reduce_data(self):
        """Step 1: Reduce time series using FA if not exists."""
        print("Step 1: Reducing time series using Factor Analysis...")
        
        # Check if reduced files already exist
        if os.listdir(self.reduced_dir):
            print("Reduced files already exist, skipping reduction.")
            return
        
        self.loader.reduce_all_files(self.reduced_dir)
        print("✓ Data reduction completed.")
    
    def train_model(self, model_type: str = 'ar', **kwargs):
        """Step 2: Train forecasting model."""
        print(f"Step 2: Training {model_type.upper()} model...")
        
        # Initialize model
        self.model = TimeSeriesForecastingModel(model_type=model_type, **kwargs)
        
        # Train model on each file dynamically
        file_count = 0
        for person_id, filename, df in tqdm(self.loader.iter_split_dfs('train')):
            reduced_path = os.path.join(self.reduced_dir, f"{os.path.splitext(filename)[0]}_fa1.csv")
            if os.path.exists(reduced_path):
                reduced_df = self.loader.load_reduced_file(reduced_path)
                self.model.fit(reduced_df)
                file_count += 1
        
        if file_count == 0:
            raise ValueError("No training data found")
        
        # Save model
        model_path = os.path.join(self.model_dir, f"{model_type}_model.joblib")
        self.model.save_model(model_path)
        print(f"✓ Model trained on {file_count} files and saved to {model_path}")
    
    def evaluate_model(self, model_type: str = 'ar'):
        """Step 3: Evaluate model performance."""
        print("Step 3: Evaluating model performance...")
        
        # Load model if not already loaded
        if self.model is None:
            model_path = os.path.join(self.model_dir, f"{model_type}_model.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            self.model = TimeSeriesForecastingModel(model_type=model_type)
            self.model.load_model(model_path)
        
        # Evaluate on test data dynamically
        all_metrics = []
        file_count = 0
        total_samples = 0
        for person_id, filename, df in self.loader.iter_split_dfs('test'):
            reduced_path = os.path.join(self.reduced_dir, f"{os.path.splitext(filename)[0]}_fa1.csv")
            if os.path.exists(reduced_path):
                test_df = self.loader.load_reduced_file(reduced_path)

                total_samples += test_df.shape[0]
                metrics = self.model.evaluate(test_df)
                if metrics:  # Only add if metrics were calculated
                    all_metrics.append(metrics)
                file_count += 1
        print(total_samples, 'total_samples')
        # Aggregate metrics
        if all_metrics:
            agg_metrics = {}
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                agg_metrics[f'{key}_mean'] = np.mean(values)
                agg_metrics[f'{key}_std'] = np.std(values)
            
            print(f"✓ Evaluation completed on {file_count} test files ({len(all_metrics)} with valid metrics)")
            print("Performance Metrics:")
            for key, value in agg_metrics.items():
                if 'mse' in key:
                    print(f"  {key}: {value:.4f}")
            
            return agg_metrics
        
        return {}


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Time Series Forecasting Pipeline")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing CSV files')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory to save models')
    parser.add_argument('--splits_file', type=str, required=True, help='JSON file with train/val/test splits')
    parser.add_argument('--reduced_dir', type=str, required=True, help='Directory for reduced 1D data')
    parser.add_argument('--model_type', type=str, default='ar', choices=['ar', 'arima', 'var', 'sgd_ar'], help='Model type')
    parser.add_argument('--forecast_horizon', type=int, default=10, help='Forecast horizon')
    parser.add_argument('--lag_order', type=int, default=10, help='Lag order for AR models')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TimeSeriesForecastingPipeline(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        splits_file=args.splits_file,
        reduced_dir=args.reduced_dir
    )
    
    # Run pipeline
    pipeline.reduce_data()
    pipeline.train_model(
        model_type=args.model_type,
        forecast_horizon=args.forecast_horizon,
        lag_order=args.lag_order
    )
    pipeline.evaluate_model(args.model_type)


if __name__ == "__main__":
    main()