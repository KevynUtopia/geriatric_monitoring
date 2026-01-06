import pandas as pd
import numpy as np
import os
import glob
import re
import json
from datetime import datetime
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from scipy.stats import chi2
import joblib
from collections import defaultdict
import sys
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FATrainer:
    def __init__(self, data_directory, model_output_directory, min_csv_length=500, splits_file=None, iterative_drop_experiment=False):
        """
        Initialize the Factor Analysis trainer.
        
        Args:
            data_directory (str): Directory containing CSV files
            model_output_directory (str): Directory to save trained models
            min_csv_length (int): Minimum number of rows required in CSV files
            splits_file (str): Path to JSON file containing train/val/test splits
        """
        self.data_directory = data_directory
        self.model_output_directory = model_output_directory
        self.min_csv_length = min_csv_length
        self.splits_file = splits_file
        self.col_names = ['timestamp', 'drink', 'eat', 'activity', 'sleep', 'social', 'sit', 'stand', 'watch (TV)']
        
        # Create output directory if it doesn't exist
        os.makedirs(model_output_directory, exist_ok=True)
        
        # Initialize CSV files for collecting KMO and Bartlett test results
        self.kmo_csv_path = os.path.join(model_output_directory, 'kmo_results.csv')
        self.bartlett_csv_path = os.path.join(model_output_directory, 'bartlett_results.csv')
        # Iterative experiment CSVs
        self.kmo_iter_csv_path = os.path.join(model_output_directory, 'kmo_results_iterative.csv')
        self.bartlett_iter_csv_path = os.path.join(model_output_directory, 'bartlett_results_iterative.csv')
        self.iterative_drop_experiment = iterative_drop_experiment
        
        # Initialize CSV files with headers if they don't exist
        self._initialize_csv_files()
        
        # Load splits if provided
        self.splits_data = None
        if splits_file and os.path.exists(splits_file):
            self.splits_data = self._load_splits(splits_file)
    
    def _initialize_csv_files(self):
        """
        Initialize CSV files with headers if they don't exist.
        """
        # Initialize KMO results CSV
        if not os.path.exists(self.kmo_csv_path):
            kmo_df = pd.DataFrame(columns=['person_id', 'kmo_all', 'kmo_model', 'timestamp'])
            kmo_df.to_csv(self.kmo_csv_path, index=False)
            print(f"Created KMO results CSV: {self.kmo_csv_path}")
        
        # Initialize Bartlett results CSV
        if not os.path.exists(self.bartlett_csv_path):
            bartlett_df = pd.DataFrame(columns=['person_id', 'chi2_statistic', 'p_value', 'timestamp'])
            bartlett_df.to_csv(self.bartlett_csv_path, index=False)
            print(f"Created Bartlett results CSV: {self.bartlett_csv_path}")
        
        # Initialize iterative experiment CSVs
        if not os.path.exists(self.kmo_iter_csv_path):
            kmo_iter_df = pd.DataFrame(columns=['person_id', 'retained_dims', 'dropped_feature', 'kmo_all', 'kmo_model', 'timestamp'])
            kmo_iter_df.to_csv(self.kmo_iter_csv_path, index=False)
            print(f"Created iterative KMO results CSV: {self.kmo_iter_csv_path}")
        if not os.path.exists(self.bartlett_iter_csv_path):
            bartlett_iter_df = pd.DataFrame(columns=['person_id', 'retained_dims', 'dropped_feature', 'chi2_statistic', 'p_value', 'timestamp'])
            bartlett_iter_df.to_csv(self.bartlett_iter_csv_path, index=False)
            print(f"Created iterative Bartlett results CSV: {self.bartlett_iter_csv_path}")
    
    def _save_kmo_result(self, person_id, kmo_all, kmo_model):
        """
        Save KMO test result to CSV file.
        
        Args:
            person_id (str): Person identifier
            kmo_all (float): Overall KMO value
            kmo_model (float): Model KMO value
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = pd.DataFrame({
            'person_id': [person_id],
            'kmo_all': [kmo_all],
            'kmo_model': [kmo_model],
            'timestamp': [timestamp]
        })
        
        # Append to existing CSV
        new_row.to_csv(self.kmo_csv_path, mode='a', header=False, index=False)
    
    def _save_bartlett_result(self, person_id, chi2_stat, p_value):
        """
        Save Bartlett test result to CSV file.
        
        Args:
            person_id (str): Person identifier
            chi2_stat (float): Chi-square statistic
            p_value (float): P-value
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = pd.DataFrame({
            'person_id': [person_id],
            'chi2_statistic': [chi2_stat],
            'p_value': [p_value],
            'timestamp': [timestamp]
        })
        
        # Append to existing CSV
        new_row.to_csv(self.bartlett_csv_path, mode='a', header=False, index=False)
        
    def _save_iterative_kmo_result(self, person_id, retained_dims, dropped_feature, kmo_all, kmo_model):
        """
        Save iterative KMO result with retained dimension count.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = pd.DataFrame({
            'person_id': [person_id],
            'retained_dims': [retained_dims],
            'dropped_feature': [dropped_feature if dropped_feature is not None else ''],
            'kmo_all': [kmo_all],
            'kmo_model': [kmo_model],
            'timestamp': [timestamp]
        })
        new_row.to_csv(self.kmo_iter_csv_path, mode='a', header=False, index=False)

    def _save_iterative_bartlett_result(self, person_id, retained_dims, dropped_feature, chi2_stat, p_value):
        """
        Save iterative Bartlett result with retained dimension count.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = pd.DataFrame({
            'person_id': [person_id],
            'retained_dims': [retained_dims],
            'dropped_feature': [dropped_feature if dropped_feature is not None else ''],
            'chi2_statistic': [chi2_stat],
            'p_value': [p_value],
            'timestamp': [timestamp]
        })
        new_row.to_csv(self.bartlett_iter_csv_path, mode='a', header=False, index=False)
    
    def _save_fa_loadings(self, person_id, fa_model, activity_columns, n_factors):
        """
        Save FA loadings to CSV file.
        
        Args:
            person_id (str): Person identifier
            fa_model: Trained FactorAnalysis model
            activity_columns (list): List of variable names
            n_factors (int): Number of factors
        """
        if fa_model is None:
            return
        
        # Calculate loadings: components_.T gives loadings matrix
        # Shape: (n_features, n_components)
        loadings_matrix = fa_model.components_.T
        
        # Create DataFrame with variable names as index and factors as columns
        loadings_df = pd.DataFrame(
            loadings_matrix,
            columns=[f"Factor{i+1}" for i in range(n_factors)],
            index=activity_columns
        )
        
        # Add person_id column
        loadings_df.insert(0, 'person_id', person_id)
        
        # Save to CSV
        loadings_csv_path = os.path.join(self.model_output_directory, f"fa_loadings_{person_id}.csv")
        loadings_df.to_csv(loadings_csv_path, index=True)
        print(f"✓ Saved loadings for {person_id}: {loadings_csv_path}")
        
    def _load_splits(self, splits_file):
        """
        Load train/val/test splits from JSON file.
        
        Args:
            splits_file (str): Path to the splits JSON file
            
        Returns:
            dict: Loaded splits data
        """
        try:
            with open(splits_file, 'r') as f:
                splits_data = json.load(f)
                print(f"✓ Loaded splits from: {splits_file}")
                return splits_data
        except Exception as e:
            print(f"✗ Error loading splits from {splits_file}: {e}")
            return None
        
    def _extract_person_id_and_date(self, filename):
        """
        Extract person ID and date from filename.
        Supports formats like:
        - ts_recording_2019_06_22_9_20_am_p_5.csv
        - p_1.csv
        
        Args:
            filename (str): Name of the CSV file
            
        Returns:
            tuple: (person_id, date_info) or (person_id, None) if no date
        """
        # Remove .csv extension
        basename = filename.replace('.csv', '')
        
        # Pattern 1: ts_recording_YYYY_MM_DD_H_MM_XM_p_N.csv
        pattern1 = r'ts_recording_(\d{4}_\d{2}_\d{2}_\d+_\d+_[ap]m)_p_(\d+)'
        match1 = re.search(pattern1, basename)
        if match1:
            date_info = match1.group(1)
            person_id = f"p_{match1.group(2)}"
            return person_id, date_info
        
        # Pattern 2: p_N.csv (already concatenated files)
        pattern2 = r'^p_(\d+)$'
        match2 = re.search(pattern2, basename)
        if match2:
            person_id = f"p_{match2.group(1)}"
            return person_id, None
            
        # Pattern 3: Other formats with p_N
        pattern3 = r'p_(\d+)'
        match3 = re.search(pattern3, basename)
        if match3:
            person_id = f"p_{match3.group(1)}"
            # Try to extract date if present
            date_match = re.search(r'(\d{4}_\d{2}_\d{2})', basename)
            date_info = date_match.group(1) if date_match else None
            return person_id, date_info
            
        return None, None
    
    def _parse_timestamp(self, x):
        """
        Parse timestamp from hhmmss format to datetime.
        
        Args:
            x: Timestamp in hhmmss format
            
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
            extra_minutes = seconds // 60
            seconds = seconds % 60
            minutes += extra_minutes
        
        if minutes >= 60:
            extra_hours = minutes // 60
            minutes = minutes % 60
            hours += extra_hours
        
        # Ensure hours don't exceed 23
        hours = hours % 24
        
        try:
            return datetime.strptime(f"2019-07-01 {hours:02d}:{minutes:02d}:{seconds:02d}", "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            print(f"Warning: Invalid timestamp {x}, using default timestamp")
            return datetime.strptime("2019-07-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    
    def _load_and_process_csv(self, csv_path):
        """
        Load and process a single CSV file.
        
        Args:
            csv_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Processed dataframe or None if file is too short
        """
        try:
            # Load the CSV
            df = pd.read_csv(csv_path, header=0)
            
            # Check minimum length
            if len(df) < self.min_csv_length:
                print(f"Skipping {os.path.basename(csv_path)}: Only {len(df)} rows (minimum {self.min_csv_length} required)")
                return None
                
            print(f"Loading {os.path.basename(csv_path)}: {len(df)} rows")
            
            # Check if timestamps are already in seconds format
            sample_timestamp = df['timestamp'].iloc[0]
            
            if pd.api.types.is_numeric_dtype(df['timestamp']):
                if sample_timestamp > 86400:  # Unix timestamp
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                else:  # Seconds from start of day
                    base_date = pd.to_datetime('2019-07-01')
                    df['timestamp'] = base_date + pd.to_timedelta(df['timestamp'], unit='s')
            else:
                # Convert hhmmss format
                df['timestamp'] = df['timestamp'].apply(self._parse_timestamp)
            
            # Get activity columns (excluding timestamp)
            activity_columns = [col for col in df.columns if col != 'timestamp']
            
            # Smooth the data
            for col in activity_columns:
                if col == 'activity':
                    df[col] = 1.0 * (df[col] > 0)
                df[col] = gaussian_filter1d(df[col], sigma=3)
            
            return df
            
        except Exception as e:
            print(f"Error loading {csv_path}: {str(e)}")
            return None
    
    def _concatenate_training_data(self, person_id, training_files):
        """
        Concatenate training data for a single person from multiple files.
        
        Args:
            person_id (str): Person identifier
            training_files (list): List of training file entries from splits
            
        Returns:
            pd.DataFrame: Concatenated and sorted dataframe
        """
        all_dfs = []
        
        print(f"Loading training files for {person_id}:")
        # Load all training files for this person (already sorted chronologically)
        for file_entry in training_files:
            filepath = file_entry['path']
            filename = file_entry['filename']
            date_info = file_entry.get('date_info', None)
            
            # find the file 
            filepath = os.path.join(self.data_directory, filepath.split('/')[-1])
            
            print(f"  - {filename} (rows: {file_entry['rows']}, date: {date_info})")
            
            df = self._load_and_process_csv(filepath)
            if df is not None:
                # Add file metadata for tracking
                df['source_file'] = filename
                if date_info:
                    df['file_date'] = date_info
                all_dfs.append(df)
        
        if not all_dfs:
            return None
            
        # Concatenate all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Sort by timestamp to ensure chronological order
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove metadata columns
        metadata_cols = ['source_file', 'file_date']
        for col in metadata_cols:
            if col in combined_df.columns:
                combined_df = combined_df.drop(col, axis=1)
        
        print(f"✓ Concatenated training data: {len(combined_df)} total rows from {len(training_files)} files")
        return combined_df
    
    def _train_factor_analysis_model(self, df, person_id, FA=False):
        """
        Train a Factor Analysis model for a person's data.
        
        Args:
            df (pd.DataFrame): The person's concatenated training data
            person_id (str): Person identifier
            
        Returns:
            tuple: (model, n_factors, scaler) or None if training failed
        """
        try:
            # Get activity columns (excluding timestamp)
            activity_columns = [col for col in df.columns if col != 'timestamp']
            X = df[activity_columns].values
            
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Use PCA to determine optimal number of factors
            pca = PCA()
            pca.fit(X_scaled)
            n_factors = sum(pca.explained_variance_ > 1)  # Kaiser criterion
            n_factors = max(1, min(n_factors, len(activity_columns)))  # Ensure valid range
            
            print(f"Person {person_id}: Using {n_factors} factors (Kaiser criterion)")

            kmo_all, kmo_model = calculate_kmo(X_scaled)
            print(f"KMO value (should be >0.6): {kmo_model}")
            
            chi2, p_value = calculate_bartlett_sphericity(X_scaled)
            print(f"Bartlett's test p-value (should be <0.05): {p_value}")
            # exit(0)
            
            # Save KMO and Bartlett test results to CSV files
            self._save_kmo_result(person_id, kmo_all, kmo_model)
            self._save_bartlett_result(person_id, chi2, p_value)
    
            
            # Train Factor Analysis model
            if FA:
                fa = FactorAnalysis(n_components=n_factors, random_state=42)
                fa.fit(X_scaled)
                # Save FA loadings to CSV
                self._save_fa_loadings(person_id, fa, activity_columns, n_factors)
            else:
                fa = None
            exit(0)
            # Optional: run iterative feature-drop experiment and archive metrics
            if self.iterative_drop_experiment:
                self._run_iterative_drop_experiment(X_scaled, activity_columns, person_id)

            # Save the model
            model_filename = f"fa_model_{person_id}.joblib"
            model_path = os.path.join(self.model_output_directory, model_filename)
            
            # Save model, scaler, and metadata
            model_data = {
                'fa_model': fa,
                'scaler': scaler,
                'n_factors': n_factors,
                'activity_columns': activity_columns,
                'training_samples': len(df),
                'person_id': person_id,
                'trained_on_split': 'train'  # Indicate this was trained on training split
            }
            
            joblib.dump(model_data, model_path)
            print(f"✓ Saved model for {person_id}: {model_path}")
            
            return fa, n_factors, scaler
            
        except Exception as e:
            print(f"✗ Error training model for {person_id}: {str(e)}")
            return None

    def _run_iterative_drop_experiment(self, X_scaled, activity_columns, person_id):
        """
        Iteratively drop the highest-loading feature and recompute FA, KMO, and Bartlett,
        archiving cues for each retained-dimension count down to 1.
        """
        try:
            current_columns = list(activity_columns)
            current_X = X_scaled.copy()
            last_dropped = None

            while True:
                retained_dims = len(current_columns)
                # Archive KMO and Bartlett for current set
                kmo_all, kmo_model = calculate_kmo(current_X)
                chi2_stat, p_value = calculate_bartlett_sphericity(current_X)
                self._save_iterative_kmo_result(person_id, retained_dims, last_dropped, kmo_all, kmo_model)
                self._save_iterative_bartlett_result(person_id, retained_dims, last_dropped, chi2_stat, p_value)

                print(f"[{person_id}] Iterative drop: retained_dims={retained_dims}, last_dropped={last_dropped if last_dropped else 'None'}, KMO={kmo_model:.4f}, p={p_value:.4g}")

                if retained_dims == 1:
                    break

                # Determine number of factors for current set via PCA (Kaiser criterion)
                pca = PCA()
                pca.fit(current_X)
                n_factors = sum(pca.explained_variance_ > 1)
                n_factors = max(1, min(n_factors, retained_dims))

                # Fit FA and compute per-feature importance via max absolute loading across components
                fa = FactorAnalysis(n_components=n_factors, random_state=42)
                fa.fit(current_X)
                # components_ shape: (n_components, n_features). Compute per-feature max |loading|
                loadings = fa.components_  # (k, d)
                per_feature_strength = np.max(np.abs(loadings), axis=0)  # (d,)

                # Determine how many to drop this iteration (drop up to 5, keep at least 1)
                num_to_drop = min(5, retained_dims - 1)
                # Pick indices of top-importance features
                drop_indices = np.argsort(-per_feature_strength)[:num_to_drop]
                drop_indices = sorted(drop_indices.tolist(), reverse=True)  # reverse sort to delete safely
                dropped_names = [current_columns[i] for i in sorted(drop_indices)]
                last_dropped = ", ".join(dropped_names)

                # Remove the selected features
                for idx in drop_indices:
                    current_columns.pop(idx)
                current_X = np.delete(current_X, drop_indices, axis=1)

        except Exception as e:
            print(f"✗ Iterative drop experiment failed for {person_id}: {str(e)}")
    
    def train_models_with_splits(self):
        """
        Train Factor Analysis models using the train/val/test splits.
        Only uses training data for each person.
        
        Returns:
            dict: Dictionary with training results for each person
        """
        if not self.splits_data:
            print("✗ No splits data loaded. Please provide a splits file.")
            return {}
        
        print("="*80)
        print("TRAINING FACTOR ANALYSIS MODELS WITH SPLITS")
        print("="*80)
        
        persons_data = self.splits_data.get('persons', {})
        if not persons_data:
            print("✗ No person data found in splits.")
            return {}
        
        print(f"Found {len(persons_data)} persons in splits data")
        
        # Train models for each person using only their training data
        training_results = {}
        
        # Sort persons by ID for consistent output
        sorted_persons = sorted(persons_data.keys(), key=lambda x: int(x.split('_')[1]))
        
        for person_id in sorted_persons:
            person_splits = persons_data[person_id]
            training_files = person_splits.get('train', {}).get('files', [])
            
            print(f"\n{'='*60}")
            print(f"Training model for {person_id}")
            print(f"{'='*60}")
            
            if not training_files:
                print(f"✗ No training files found for {person_id}")
                training_results[person_id] = {'success': False, 'error': 'No training files'}
                continue
            
            print(f"Using {len(training_files)} training files")
            
            # Concatenate training data for this person
            person_df = self._concatenate_training_data(person_id, training_files)
            
            if person_df is not None:
                # Train Factor Analysis model
                model_result = self._train_factor_analysis_model(person_df, person_id, FA=self.iterative_drop_experiment or True)
                
                if model_result:
                    fa_model, n_factors, scaler = model_result
                    training_results[person_id] = {
                        'model_path': os.path.join(self.model_output_directory, f"fa_model_{person_id}.joblib"),
                        'n_factors': n_factors,
                        'training_samples': len(person_df),
                        'n_training_files': len(training_files),
                        'success': True
                    }
                    print(f"✓ Successfully trained model for {person_id}")
                else:
                    training_results[person_id] = {'success': False, 'error': 'Model training failed'}
                    print(f"✗ Failed to train model for {person_id}")
            else:
                training_results[person_id] = {'success': False, 'error': 'No valid training data found'}
                print(f"✗ No valid training data found for {person_id}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY")
        print(f"{'='*80}")
        
        successful = sum(1 for result in training_results.values() if result.get('success', False))
        total = len(training_results)
        
        print(f"Total persons: {total}")
        print(f"Successful models: {successful}")
        print(f"Failed models: {total - successful}")
        
        for person_id, result in training_results.items():
            if result.get('success', False):
                print(f"✓ {person_id}: {result['n_factors']} factors, "
                      f"{result['training_samples']} samples, {result['n_training_files']} training files")
            else:
                print(f"✗ {person_id}: {result.get('error', 'Unknown error')}")
        
        return training_results

    def train_all_models(self):
        """
        Main method to train Factor Analysis models for all persons.
        Uses splits if available, otherwise falls back to all files.
        
        Returns:
            dict: Dictionary with training results for each person
        """
        if self.splits_data:
            print("Using train/val/test splits for training")
            return self.train_models_with_splits()
        else:
            print("No splits provided, using all available files")
            return self._train_all_models_legacy()
    
    def _train_all_models_legacy(self):
        """
        Legacy method to train on all available files (no splits).
        """
        # Find all CSV files
        csv_pattern = os.path.join(self.data_directory, "*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"No CSV files found in {self.data_directory}")
            return {}
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Group files by person ID
        person_files = defaultdict(list)
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            person_id, date_info = self._extract_person_id_and_date(filename)
            
            if person_id:
                person_files[person_id].append((csv_file, date_info))
            else:
                print(f"Warning: Could not extract person ID from {filename}")
        
        print(f"Found data for {len(person_files)} unique persons: {sorted(person_files.keys())}")
        
        # Train models for each person
        training_results = {}
        
        for person_id, files in person_files.items():
            print(f"\n{'='*60}")
            print(f"Training model for {person_id}")
            print(f"{'='*60}")
            
            # Sort files by date if date info is available
            files.sort(key=lambda x: x[1] if x[1] else "")
            
            print(f"Files for {person_id}:")
            for filepath, date_info in files:
                print(f"  - {os.path.basename(filepath)} ({date_info if date_info else 'no date info'})")
            
            # Concatenate data for this person (legacy method)
            person_df = self._concatenate_person_data_legacy(files)
            
            if person_df is not None:
                # Train Factor Analysis model
                model_result = self._train_factor_analysis_model(person_df, person_id, FA=self.iterative_drop_experiment or True)
                
                if model_result:
                    fa_model, n_factors, scaler = model_result
                    training_results[person_id] = {
                        'model_path': os.path.join(self.model_output_directory, f"fa_model_{person_id}.joblib"),
                        'n_factors': n_factors,
                        'training_samples': len(person_df),
                        'n_files': len(files),
                        'success': True
                    }
                    print(f"✓ Successfully trained model for {person_id}")
                else:
                    training_results[person_id] = {'success': False, 'error': 'Model training failed'}
                    print(f"✗ Failed to train model for {person_id}")
            else:
                training_results[person_id] = {'success': False, 'error': 'No valid data found'}
                print(f"✗ No valid data found for {person_id}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY")
        print(f"{'='*80}")
        
        successful = sum(1 for result in training_results.values() if result.get('success', False))
        total = len(training_results)
        
        print(f"Total persons: {total}")
        print(f"Successful models: {successful}")
        print(f"Failed models: {total - successful}")
        
        for person_id, result in training_results.items():
            if result.get('success', False):
                print(f"✓ {person_id}: {result['n_factors']} factors, "
                      f"{result['training_samples']} samples, {result['n_files']} files")
            else:
                print(f"✗ {person_id}: {result.get('error', 'Unknown error')}")
        
        return training_results
    
    def _concatenate_person_data_legacy(self, person_files):
        """
        Legacy method to concatenate data for a single person from multiple files.
        
        Args:
            person_files (list): List of (filepath, date_info) tuples for one person
            
        Returns:
            pd.DataFrame: Concatenated and sorted dataframe
        """
        all_dfs = []
        
        # Load all files for this person
        for filepath, date_info in person_files:
            print(filepath)
            exit(0)
            df = self._load_and_process_csv(filepath)
            if df is not None:
                # Add date info for sorting if available
                if date_info:
                    df['file_date'] = date_info
                all_dfs.append(df)
        
        if not all_dfs:
            return None
            
        # Concatenate all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Sort by timestamp to ensure chronological order
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove file_date column if it exists
        if 'file_date' in combined_df.columns:
            combined_df = combined_df.drop('file_date', axis=1)
        
        print(f"Concatenated data: {len(combined_df)} total rows")
        return combined_df

# Example usage and CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Factor Analysis models for person-specific data')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing CSV files')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory to save trained models')
    parser.add_argument('--splits_file', type=str, default=None,
                        help='JSON file containing train/val/test splits')
    parser.add_argument('--min_length', type=int, default=500,
                        help='Minimum number of rows required in CSV files')
    parser.add_argument('--iterative_drop_experiment', action='store_true',
                        help='Enable iterative highest-loading feature drop experiment with archiving')
    
    args = parser.parse_args()
    
    # Create trainer and train models
    trainer = FATrainer(
        data_directory=args.data_dir,
        model_output_directory=args.model_dir,
        min_csv_length=args.min_length,
        splits_file=args.splits_file,
        iterative_drop_experiment=args.iterative_drop_experiment
    )
    
    results = trainer.train_all_models()
    
    print(f"\nModels saved to: {args.model_dir}")
    print("Training completed!")
