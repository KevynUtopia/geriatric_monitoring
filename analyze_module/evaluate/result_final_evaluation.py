import pandas as pd
import numpy as np
import os
import glob
import re
import json
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix


class HumanSystemEvaluator:
    def __init__(self, system_output_dir, human_evaluation_dir, output_dir, data_splits_path, video_time_periods_path=None, camera_to_person_mapping=None):
        """
        Initialize the evaluator for comparing human evaluation results with system outputs.
        
        Args:
            system_output_dir (str): Directory containing system output files (ts_recording_*_p_*.csv)
            human_evaluation_dir (str): Directory containing human evaluation subfolders (recording_*/p_*.csv)
            output_dir (str): Directory to save evaluation results
            data_splits_path (str): Path to data_splits.json file
            video_time_periods_path (str): Path to video_time_periods.csv file for filtering
            camera_to_person_mapping (dict): Mapping from camera_id to person_id (e.g., {'cam_10': 'p_1', 'cam_11': 'p_2'})
        """
        self.system_output_dir = system_output_dir
        self.human_evaluation_dir = human_evaluation_dir
        self.output_dir = output_dir
        self.data_splits_path = data_splits_path
        self.video_time_periods_path = video_time_periods_path
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up camera to person mapping
        if camera_to_person_mapping is None:
            # Default mapping: cam_X -> p_(X-9) for cam_10,11,12,13,... -> p_1,2,3,4,...
            self.camera_to_person_mapping = {
                f'cam_{i}': f'p_{i-9}' for i in range(10, 25)
            }
        else:
            self.camera_to_person_mapping = camera_to_person_mapping
        
        # Create reverse mapping for convenience
        self.person_to_camera_mapping = {v: k for k, v in self.camera_to_person_mapping.items()}
        
        # Load data splits
        self.data_splits = self._load_data_splits()
        
        # Load video time periods for filtering
        self.video_time_periods = self._load_video_time_periods()
        
        # Print camera to person mapping being used
        if self.video_time_periods:
            print(f"Using camera-to-person mapping: {dict(list(self.camera_to_person_mapping.items())[:5])}{'...' if len(self.camera_to_person_mapping) > 5 else ''}")
            print("Video time period filtering: ENABLED - only timestamps within recorded video periods will be evaluated")
        
        # Storage for discovered files
        self.human_files_by_identity = defaultdict(list)  # {identity: [(recording_session, file_path), ...]}
        self.system_files_by_identity = defaultdict(list)  # {identity: [(recording_session, file_path), ...]}
        self.matched_pairs_by_identity = defaultdict(list)  # {identity: [(sys_file, human_file, session), ...]}
        
        # Storage organized by splits
        self.matched_pairs_by_split = {
            'train': defaultdict(list),  # {identity: [(sys_file, human_file, session), ...]}
            'val': defaultdict(list),
            'test': defaultdict(list)
        }
        
        # Storage for evaluation results
        self.identity_results = {}  # {split: {identity: metrics_dict}}
        self.summary_results = {}   # {split: {micro_metrics, macro_metrics}}

    def _load_video_time_periods(self):
        """
        Load video time periods from CSV file for filtering timestamps.
        
        Returns:
            dict: Nested dictionary {recording_session: {person_id: [(start_sec, end_sec), ...]}}
        """
        if not self.video_time_periods_path or not os.path.exists(self.video_time_periods_path):
            print("Warning: Video time periods file not found. No timestamp filtering will be applied.")
            return {}
        
        try:
            df = pd.read_csv(self.video_time_periods_path)
            print(f"Loaded video time periods from {os.path.basename(self.video_time_periods_path)}")
            
            # Organize by recording session and person
            periods = defaultdict(lambda: defaultdict(list))
            
            for _, row in df.iterrows():
                recording_session = row['Recording Day']
                camera_id = row['Camera ID']
                start_seconds = row['Actual Start (seconds)']
                end_seconds = row['Actual End (seconds)']
                
                # Map camera ID to person ID
                person_id = self.camera_to_person_mapping.get(camera_id)
                if person_id:
                    periods[recording_session][person_id].append((start_seconds, end_seconds))
            
            # Sort time periods for each person in each session
            for recording_session in periods:
                for person_id in periods[recording_session]:
                    periods[recording_session][person_id].sort()
            
            # Print summary
            total_sessions = len(periods)
            total_periods = sum(len(person_periods) for session_periods in periods.values() 
                              for person_periods in session_periods.values())
            print(f"Processed {total_periods} video periods across {total_sessions} recording sessions")
            
            return periods
            
        except Exception as e:
            print(f"Error loading video time periods: {e}")
            return {}

    def _is_timestamp_in_recorded_periods(self, recording_session, person_id, timestamp):
        """
        Check if a timestamp falls within any recorded video period for the given recording session and person.
        
        Args:
            recording_session (str): Recording session name (e.g., 'recording_2019_06_22_9_20_am')
            person_id (str): Person ID (e.g., 'p_1')
            timestamp (float): Timestamp in seconds from midnight
            
        Returns:
            bool: True if timestamp is within a recorded period, False otherwise
        """
        if not self.video_time_periods:
            return True  # No filtering if no video periods loaded
        
        if recording_session not in self.video_time_periods:
            return False  # No recorded periods for this session
        
        if person_id not in self.video_time_periods[recording_session]:
            return False  # No recorded periods for this person in this session
        
        # Check if timestamp falls within any recorded period
        periods = self.video_time_periods[recording_session][person_id]
        for start_time, end_time in periods:
            if start_time <= timestamp <= end_time:
                return True
        
        return False

    def _filter_timestamps_by_recorded_periods(self, recording_session, person_id, timestamps, values):
        """
        Filter timestamps and corresponding values to only include those within recorded video periods.
        
        Args:
            recording_session (str): Recording session name
            person_id (str): Person ID
            timestamps (np.array): Array of timestamps
            values (np.array): Array of values corresponding to timestamps
            
        Returns:
            tuple: (filtered_timestamps, filtered_values)
        """
        if not self.video_time_periods:
            return timestamps, values  # No filtering if no video periods loaded
        
        original_count = len(timestamps)
        
        # Create mask for timestamps within recorded periods
        mask = np.array([
            self._is_timestamp_in_recorded_periods(recording_session, person_id, ts)
            for ts in timestamps
        ])
        
        if not np.any(mask):
            return np.array([]), np.array([])  # No valid timestamps
        
        filtered_timestamps = timestamps[mask]
        filtered_values = values[mask]
        filtered_count = len(filtered_timestamps)
        
        # Only log if significant filtering occurred
        if original_count > 0 and filtered_count < original_count:
            retention_rate = filtered_count / original_count
            print(f"      Filtered {person_id} in {recording_session}: {filtered_count}/{original_count} timestamps retained ({retention_rate:.1%})")
        
        return filtered_timestamps, filtered_values

    def _load_data_splits(self):
        """Load the data splits from JSON file."""
        try:
            with open(self.data_splits_path, 'r') as f:
                data_splits = json.load(f)
            print(f"Loaded data splits from {os.path.basename(self.data_splits_path)}")
            
            # Print split summary
            if 'persons' in data_splits:
                total_persons = len(data_splits['persons'])
                train_count = sum(1 for person_data in data_splits['persons'].values() 
                                if 'train' in person_data and person_data['train']['files'])
                val_count = sum(1 for person_data in data_splits['persons'].values() 
                              if 'val' in person_data and person_data['val']['files'])
                test_count = sum(1 for person_data in data_splits['persons'].values() 
                               if 'test' in person_data and person_data['test']['files'])
                print(f"Split configuration: {train_count} train, {val_count} val, {test_count} test persons")
            
            return data_splits
        except Exception as e:
            print(f"Error loading data splits: {e}")
            return None
    
    def _get_split_for_file(self, recording_session, person_id):
        """
        Determine which split (train/val/test) a file belongs to.
        
        Args:
            recording_session (str): Recording session name (e.g., 'recording_2019_06_22_9_20_am')
            person_id (str): Person ID (e.g., 'p_1')
            
        Returns:
            str: 'train', 'val', 'test', or None if not found
        """
        if not self.data_splits or 'persons' not in self.data_splits:
            return None
            
        if person_id not in self.data_splits['persons']:
            return None
            
        person_data = self.data_splits['persons'][person_id]
        
        # Check each split
        for split_name in ['train', 'val', 'test']:
            if split_name in person_data and 'files' in person_data[split_name]:
                files_in_split = person_data[split_name]['files']
                for file_info in files_in_split:
                    # More precise matching: check if the exact recording session matches
                    filename = file_info['filename']
                    expected_filename = f"ts_{recording_session}_{person_id}.csv"
                    if filename == expected_filename:
                        return split_name
        
        return None
    
    def extract_recording_info_from_system_file(self, filename):
        """
        Extract recording session and person ID from system output filename.
        
        Args:
            filename (str): System output filename (e.g., "ts_recording_2019_06_22_9_20_am_p_1.csv")
            
        Returns:
            tuple: (recording_session, person_id) or (None, None) if parsing fails
        """
        # Pattern: ts_recording_2019_06_22_9_20_am_p_1.csv
        pattern = r'ts_(recording_\d{4}_\d{2}_\d{2}_\d+_\d+_[ap]m)_p_(\d+)\.csv'
        match = re.search(pattern, filename)
        
        if match:
            recording_session = match.group(1)  # recording_2019_06_22_9_20_am
            person_id = f"p_{match.group(2)}"   # p_1
            return recording_session, person_id
            
        return None, None
    
    
    def discover_human_evaluation_files(self):
        """
        Traverse human evaluation directory to find all evaluation files organized by identity.
        """
        print("Discovering human evaluation files...")
        
        # Find all CSV files in subdirectories
        pattern = os.path.join(self.human_evaluation_dir, "*", "*.csv")
        human_files = glob.glob(pattern)
        
        print(f"Found {len(human_files)} human evaluation files")
        
        for file_path in human_files:
            # Extract relative path from base directory
            rel_path = os.path.relpath(file_path, self.human_evaluation_dir)
            path_parts = rel_path.replace('\\', '/').split('/')
            directory = path_parts[-2]  # recording_2019_06_22_9_20_am
            person_id = path_parts[-1]
            system_file = os.path.join(self.system_output_dir, f"ts_{directory}_{person_id}")
            if not os.path.exists(system_file):
                print(f"  Warning: System file not found: {system_file}")
                continue
            
            self.human_files_by_identity[person_id].append(directory)


        # Sort files by recording session for each identity
        for person_id in self.human_files_by_identity:
            self.human_files_by_identity[person_id].sort()
        
        print(f"Organized by {len(self.human_files_by_identity)} identities")
    
    def organize_files_by_splits(self):
        """
        Organize the discovered human files by identity into train/val/test splits
        according to the data splits configuration.
        """
        print("Organizing files by train/val/test splits...")
        
        split_counts = {'train': 0, 'val': 0, 'test': 0, 'unknown': 0}
        
        for person_id, session_files in self.human_files_by_identity.items():
            # Based on debug output, we now know the structure:
            # person_id includes .csv extension (e.g., 'p_6.csv')
            # session_files is a list of recording session strings
            
            # Clean person_id (remove .csv extension if present)
            clean_person_id = person_id.replace('.csv', '')
            
            for session_info in session_files:
                if isinstance(session_info, tuple) and len(session_info) == 2:
                    # Expected structure: (recording_session, file_path)
                    recording_session, file_path = session_info
                elif isinstance(session_info, str):
                    # Actual structure: just recording session strings
                    recording_session = session_info
                    # Construct file path
                    file_path = os.path.join(self.human_evaluation_dir, recording_session, person_id)
                else:
                    print(f"Warning: Unexpected session_info structure for {person_id}: {session_info}")
                    continue
                
                # Determine which split this file belongs to
                split = self._get_split_for_file(recording_session, clean_person_id)
                
                # Check if corresponding CSV file exists in system_output_dir
                expected_system_filename = f"ts_{recording_session}_{clean_person_id}.csv"
                expected_system_path = os.path.join(self.system_output_dir, expected_system_filename)
                
                csv_exists = os.path.exists(expected_system_path)
                
                if split in ['train', 'val', 'test'] and csv_exists:
                    # Add to split-specific storage
                    self.matched_pairs_by_split[split][clean_person_id].append((None, file_path, recording_session))
                    split_counts[split] += 1
                elif split in ['train', 'val', 'test'] and not csv_exists:
                    print(f"    ⚠️  Skipping {clean_person_id} in {recording_session} ({split} split) - CSV file not found")
                else:
                    split_counts['unknown'] += 1
        
        print(f"Split distribution: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}")
        if split_counts['unknown'] > 0:
            print(f"  Warning: {split_counts['unknown']} files with unknown split")
    
    def find_matching_pairs(self):
        """
        For each human evaluation file organized by splits, find the corresponding system output file.
        
        This method performs CSV existence checking:
        - Only includes evaluations where the corresponding CSV file exists in system_output_dir
        - Skips evaluations where CSV files are missing
        - Does not count skipped evaluations in final metrics calculation
        - Provides detailed logging of which files were found vs skipped
        
        Returns:
            tuple: (total_matched, total_unmatched, skipped_entries)
                - total_matched: Number of successfully matched file pairs
                - total_unmatched: Number of skipped evaluations due to missing CSV files
                - skipped_entries: List of dictionaries with details about skipped evaluations
        """
        print("Finding corresponding system output files...")
        
        total_matched = 0
        total_unmatched = 0
        skipped_entries = []
        
        for split_name in ['train', 'val', 'test']:
            split_data = self.matched_pairs_by_split[split_name]
            
            for person_id, file_entries in list(split_data.items()):
                matched_entries = []
                
                for system_file, human_file, recording_session in file_entries:
                    # Construct expected system output filename
                    expected_system_filename = f"ts_{recording_session}_{person_id}.csv"
                    expected_system_path = os.path.join(self.system_output_dir, expected_system_filename)
                    
                    if os.path.exists(expected_system_path):
                        # Found matching system file
                        matched_entries.append((expected_system_path, human_file, recording_session))
                        total_matched += 1
                        print(f"    ✓ Found system file: {expected_system_filename}")
                    else:
                        # Skip this evaluation - CSV file doesn't exist
                        skipped_entry = {
                            'person_id': person_id,
                            'recording_session': recording_session,
                            'expected_files': [expected_system_filename],
                            'split': split_name
                        }
                        skipped_entries.append(skipped_entry)
                        total_unmatched += 1
                        print(f"    ✗ SKIPPING: System file not found: {expected_system_filename}")
                
                # Update with matched pairs (only keep those with both system and human files)
                if matched_entries:
                    split_data[person_id] = matched_entries
                    # Also update the overall matched pairs
                    self.matched_pairs_by_identity[person_id].extend(matched_entries)
                else:
                    # Remove person from split if no matches found
                    del split_data[person_id]
                    print(f"    ⚠️  Removed {person_id} from {split_name} split - no matching system files")
        
        print(f"\nFile matching summary:")
        print(f"  ✓ Matched pairs: {total_matched}")
        print(f"  ✗ Skipped evaluations: {total_unmatched}")
        
        if skipped_entries:
            print(f"\nSkipped evaluations details:")
            for entry in skipped_entries:
                print(f"    - {entry['person_id']} in {entry['recording_session']} ({entry['split']} split)")
                print(f"      Missing files: {', '.join(entry['expected_files'])}")
        
        return total_matched, total_unmatched, skipped_entries

    def print_split_summary(self):
        """
        Print a detailed summary of matched pairs organized by splits.
        """
        print("\n" + "="*50)
        print("SPLIT SUMMARY")
        print("="*50)
        
        for split_name in ['train', 'val', 'test']:
            split_data = self.matched_pairs_by_split[split_name]
            
            if split_data:
                total_pairs = sum(len(pairs) for pairs in split_data.values())
                total_identities = len(split_data)
                
                print(f"{split_name.upper()}: {total_identities} identities, {total_pairs} pairs")
            else:
                print(f"{split_name.upper()}: No data")
    
    def smart_load_system_file(self, file_path):
        """
        Load system output file and extract timestamps and predictions.
        Now supports reading separate components (anomaly_scores, variance_curve) and taking their maximum.
        
        Args:
            file_path (str): Path to system output CSV file
            
        Returns:
            tuple: (timestamps, predictions) or (None, None) if loading fails
        """
        try:
            # First, try to find the corresponding _separate.csv file
            separate_file_path = file_path.replace('.csv', '_separate.csv')
            
            if os.path.exists(separate_file_path):
                # Load the separate file with individual components
                df = pd.read_csv(separate_file_path)
                
                # Try to find time column
                time_col = None
                for col in ['seconds', 'second', 'time', 'timestamp', 'Time', 'Seconds']:
                    if col in df.columns:
                        time_col = col
                        break
                
                if time_col is None:
                    print(f"    Warning: No time column found in {os.path.basename(separate_file_path)}")
                    return None, None
                
                # Check for the required separate component columns
                if 'anomaly_scores' in df.columns and 'variance_curve' in df.columns:
                    # Clean and validate data
                    df = df.dropna(subset=[time_col, 'anomaly_scores', 'variance_curve'])
                    
                    # Convert time to numeric if needed
                    if df[time_col].dtype == 'object':
                        try:
                            # Try to parse as time format (HH:MM:SS)
                            time_series = pd.to_datetime(df[time_col], format='%H:%M:%S')
                            df[time_col] = time_series.dt.hour * 3600 + time_series.dt.minute * 60 + time_series.dt.second
                        except:
                            # Try to convert directly to numeric
                            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
                    
                    df = df.dropna(subset=[time_col, 'anomaly_scores', 'variance_curve'])
                    
                    if len(df) == 0:
                        return None, None
                    
                    # Calculate the maximum of anomaly_scores and variance_curve as final prediction
                    # final_predictions = np.minimum(df['anomaly_scores'].values, df['variance_curve'].values)
                    final_predictions = np.clip(df['anomaly_scores'].values + df['variance_curve'].values, 0, 1)
                    
                    return df[time_col].values, final_predictions
                else:
                    print(f"    Warning: Required columns 'anomaly_scores' and 'variance_curve' not found in {os.path.basename(separate_file_path)}")
            
            # Fall back to original method if separate file doesn't exist or doesn't have required columns
            df = pd.read_csv(file_path)
            
            # Try to find time column
            time_col = None
            for col in ['seconds', 'second', 'time', 'timestamp', 'Time', 'Seconds']:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col is None:
                print(f"    Warning: No time column found in {os.path.basename(file_path)}")
                return None, None
            
            # Try to find prediction/score column  
            pred_col = None
            for col in ['alarm', 'prediction', 'pred', 'score', 'anomaly', 'alert', 'Alarm', 'Prediction']:
                if col in df.columns:
                    pred_col = col
                    break
            
            if pred_col is None:
                print(f"    Warning: No prediction column found in {os.path.basename(file_path)}")
                return None, None
            
            # Clean and validate data
            df = df.dropna(subset=[time_col, pred_col])
            
            # Convert time to numeric if needed
            if df[time_col].dtype == 'object':
                try:
                    # Try to parse as time format (HH:MM:SS)
                    time_series = pd.to_datetime(df[time_col], format='%H:%M:%S')
                    df[time_col] = time_series.dt.hour * 3600 + time_series.dt.minute * 60 + time_series.dt.second
                except:
                    # Try to convert directly to numeric
                    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
            
            df = df.dropna(subset=[time_col, pred_col])
            
            if len(df) == 0:
                return None, None
                
            print(f"    Loaded combined results from {os.path.basename(file_path)} (fallback method)")
            return df[time_col].values, df[pred_col].values
            
        except Exception as e:
            print(f"    Error loading system file {os.path.basename(file_path)}: {e}")
            return None, None

    def smart_load_human_file(self, file_path):
        """
        Load human evaluation file and extract timestamps and binary labels.
        
        Args:
            file_path (str): Path to human evaluation CSV file
            
        Returns:
            tuple: (timestamps, labels) or (None, None) if loading fails
        """
        try:
            df = pd.read_csv(file_path)
            
            # Try to find time column
            time_col = None
            for col in ['second', 'seconds', 'time', 'timestamp', 'Time', 'Second']:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col is None:
                print(f"    Warning: No time column found in {os.path.basename(file_path)}")
                return None, None
            
            # Try to find results column
            result_col = None
            for col in ['results', 'result', 'label', 'ground_truth', 'gt', 'annotation', 'Results']:
                if col in df.columns:
                    result_col = col
                    break
            
            if result_col is None:
                print(f"    Warning: No result column found in {os.path.basename(file_path)}")
                return None, None
                
            # Clean and validate data
            df = df.dropna(subset=[time_col, result_col])
            
            # Convert time to numeric if needed
            if df[time_col].dtype == 'object':
                try:
                    # Try to parse as time format (HH:MM:SS)
                    time_series = pd.to_datetime(df[time_col], format='%H:%M:%S')
                    df[time_col] = time_series.dt.hour * 3600 + time_series.dt.minute * 60 + time_series.dt.second
                except:
                    # Try to convert directly to numeric
                    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
            
            df = df.dropna(subset=[time_col, result_col])
            
            if len(df) == 0:
                return None, None
            
            # Convert to binary: only values > 1 are positive, 0 and 1 are negative
            binary_labels = (df[result_col] > 1).astype(int)
            
            return df[time_col].values, binary_labels.values
            
        except Exception as e:
            print(f"    Error loading human file {os.path.basename(file_path)}: {e}")
            return None, None

    def process_into_snippets(self, timestamps, predictions, labels, snippet_duration=20):
        """
        Process time series data into fixed-duration snippets.
        
        Args:
            timestamps (np.array): Array of timestamps in seconds
            predictions (np.array): Array of system predictions
            labels (np.array): Array of human labels
            snippet_duration (int): Duration of each snippet in seconds
            
        Returns:
            tuple: (snippet_predictions, snippet_labels) or (None, None) if no valid data
        """
        if len(timestamps) == 0:
            return None, None
        
        # Find the time range
        min_time = int(timestamps.min())
        max_time = int(timestamps.max())
        
        snippet_predictions = []
        snippet_labels = []
        
        # Process data in snippet_duration second intervals
        for start_time in range(min_time, max_time + 1, snippet_duration):
            end_time = start_time + snippet_duration
            
            # Find data points in this time interval
            mask = (timestamps >= start_time) & (timestamps < end_time)
            
            if not np.any(mask):
                continue
            
            interval_predictions = predictions[mask]
            interval_labels = labels[mask]
            
            # Snippet-level prediction: use the maximum prediction in the interval
            snippet_pred = np.max(interval_predictions)
            
            # Snippet-level label: positive if ANY second in the interval is positive
            snippet_label = 1 if np.any(interval_labels > 0) else 0
            
            snippet_predictions.append(snippet_pred)
            snippet_labels.append(snippet_label)
        
        if not snippet_predictions:
            return None, None
        
        return np.array(snippet_predictions), np.array(snippet_labels)

    def load_and_align_identity_data(self, identity_pairs):
        """
        Load and align all data for a single identity across all their sessions,
        then process into 5-second snippets.
        
        Args:
            identity_pairs (list): List of (system_file, human_file, session) tuples for this identity
            
        Returns:
            tuple: (combined_predictions, combined_labels) or (None, None) if no valid data
        """
        all_predictions = []
        all_labels = []
        all_timestamps = []
        
        # Extract person_id from the first identity_pair
        if identity_pairs:
            # Get person_id from system filename (e.g., ts_recording_2019_06_22_9_20_am_p_1.csv -> p_1)
            first_system_file = identity_pairs[0][0]
            _, person_id = self.extract_recording_info_from_system_file(os.path.basename(first_system_file))
        else:
            person_id = None
        
        for system_file, human_file, session in identity_pairs:
            # Load system output
            sys_times, sys_preds = self.smart_load_system_file(system_file)
            
            # Load human evaluation
            human_times, human_labels = self.smart_load_human_file(human_file)
            
            if sys_times is None or human_times is None:
                continue
            
            # Filter system data based on recorded video periods
            if person_id and self.video_time_periods:
                sys_times, sys_preds = self._filter_timestamps_by_recorded_periods(
                    session, person_id, sys_times, sys_preds
                )
                human_times, human_labels = self._filter_timestamps_by_recorded_periods(
                    session, person_id, human_times, human_labels
                )
                
                if len(sys_times) == 0 and len(human_times) == 0:
                    print(f"    Warning: No data within recorded periods for {person_id} in {session}")
                    continue
            
            # Create system prediction lookup
            sys_dict = dict(zip(sys_times, sys_preds))
            
            # Align based on human evaluation timestamps
            session_predictions = []
            session_labels = []
            session_timestamps = []
            
            for i, timestamp in enumerate(human_times):
                # Get system prediction for this timestamp (0.0 if missing)
                pred = sys_dict.get(timestamp, 0.0)
                session_predictions.append(pred)
                session_labels.append(human_labels[i])
                session_timestamps.append(timestamp)
            
            if session_predictions:
                # Process this session into snippets
                session_pred_array = np.array(session_predictions)
                session_label_array = np.array(session_labels)
                session_time_array = np.array(session_timestamps)
                
                snippet_preds, snippet_labels = self.process_into_snippets(
                    session_time_array, session_pred_array, session_label_array
                )
                
                if snippet_preds is not None:
                    all_predictions.extend(snippet_preds)
                    all_labels.extend(snippet_labels)
        
        if not all_predictions:
            return None, None
        
        return np.array(all_predictions), np.array(all_labels)

    def calculate_binary_metrics(self, predictions, labels, thresholds=None):
        """
        Calculate binary classification metrics for given predictions and labels.
        
        Args:
            predictions (np.array): Soft predictions (0-1)
            labels (np.array): Binary labels (0 or 1)
            thresholds (list): List of thresholds to try for best F1
            
        Returns:
            dict: Dictionary of calculated metrics
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        metrics = {}
        
        # Threshold-independent metrics
        if len(np.unique(labels)) > 1:  # Need both classes for AUC
            try:
                metrics['auc'] = roc_auc_score(labels, predictions)
                metrics['average_precision'] = average_precision_score(labels, predictions)
            except:
                metrics['auc'] = 0.0
                metrics['average_precision'] = 0.0
        else:
            metrics['auc'] = 0.0
            metrics['average_precision'] = 0.0
        
        # Find best threshold based on F1 score
        best_f1 = 0.0
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in thresholds:
            binary_preds = (predictions >= threshold).astype(int)
            
            try:
                acc = accuracy_score(labels, binary_preds)
                prec = precision_score(labels, binary_preds, zero_division=0)
                rec = recall_score(labels, binary_preds, zero_division=0)
                f1 = f1_score(labels, binary_preds, zero_division=0)
                
                # Calculate confusion matrix components
                cm = confusion_matrix(labels, binary_preds)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                elif cm.shape == (1, 1):
                    # Only one class present in labels
                    if labels[0] == 0:  # All negative labels
                        tn = cm[0, 0]
                        fp = 0
                        fn = 0 
                        tp = 0
                    else:  # All positive labels
                        tn = 0
                        fp = 0
                        fn = 0
                        tp = cm[0, 0]
                else:
                    # Fallback for unexpected cases
                    tn, fp, fn, tp = 0, 0, 0, 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_metrics = {
                        'best_accuracy': acc,
                        'best_precision': prec,
                        'best_recall': rec,
                        'best_f1': f1,
                        'best_threshold': threshold,
                        'true_positives': int(tp),
                        'true_negatives': int(tn),
                        'false_positives': int(fp),
                        'false_negatives': int(fn)
                    }
            except:
                pass
        
        # Use 0.5 threshold if no best found
        if not best_metrics:
            binary_preds = (predictions >= 0.5).astype(int)
            cm = confusion_matrix(labels, binary_preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            elif cm.shape == (1, 1):
                # Only one class present in labels
                if labels[0] == 0:  # All negative labels
                    tn = cm[0, 0]
                    fp = 0
                    fn = 0 
                    tp = 0
                else:  # All positive labels
                    tn = 0
                    fp = 0
                    fn = 0
                    tp = cm[0, 0]
            else:
                # Fallback for unexpected cases
                tn, fp, fn, tp = 0, 0, 0, 0
            best_metrics = {
                'best_accuracy': accuracy_score(labels, binary_preds),
                'best_precision': precision_score(labels, binary_preds, zero_division=0),
                'best_recall': recall_score(labels, binary_preds, zero_division=0),
                'best_f1': f1_score(labels, binary_preds, zero_division=0),
                'best_threshold': 0.5,
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            }
        
        # Combine metrics
        metrics.update(best_metrics)
        
        # Add data statistics
        metrics['num_samples'] = len(predictions)  # Now referring to snippets
        metrics['num_positives'] = int(labels.sum())
        metrics['positive_rate'] = labels.mean()
        
        return metrics

    def evaluate_identity(self, identity, identity_pairs, split):
        """
        Evaluate a single identity by combining all their sessions.
        
        Args:
            identity (str): Identity name (e.g., 'p_1')
            identity_pairs (list): List of (system_file, human_file, session) tuples
            split (str): Data split name
            
        Returns:
            dict: Evaluation metrics for this identity
        """
        # Load and combine all data for this identity
        predictions, labels = self.load_and_align_identity_data(identity_pairs)
        
        if predictions is None:
            print(f"    Warning: No valid data for {identity} in {split}")
            return None
        
        # Count positive and negative samples
        num_positive = int(labels.sum())
        num_negative = len(labels) - num_positive
        total_samples = len(labels)
        positive_rate = num_positive / total_samples if total_samples > 0 else 0
        
        filter_status = " (filtered by video periods)" if self.video_time_periods else ""
        print(f"    {identity} snippet distribution: {num_positive} positive, {num_negative} negative (total: {total_samples} snippets, positive rate: {positive_rate:.3f}){filter_status}")
        
        # Calculate metrics
        metrics = self.calculate_binary_metrics(predictions, labels)
        # plot predictions and labels as two plots and place then in one colmun with two rows for visualization temprarily  
        # filder predictions for <0.5 and >0.8 away, i.e., turn them to 0
        # predictions = np.where(predictions < 0.4, 0, predictions)
        # predictions = np.where(predictions > 0.6, 0, predictions)
        # predictions = np.where(predictions != 0, 1, predictions)

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 10))
        # plt.subplot(2, 1, 1)
        # plt.plot(predictions)
        # plt.subplot(2, 1, 2)
        # plt.plot(labels)
        # plt.show()
        # exit(0)
        
        # Add identity info
        metrics['identity'] = identity
        metrics['split'] = split
        metrics['num_sessions'] = len(identity_pairs)
        
        return metrics

    def evaluate_all_identities(self):
        """
        Evaluate all identities in all splits and calculate both micro and macro metrics.
        """
        print("Calculating evaluation metrics using 5-second snippet-based approach...")
        print("  - Each 5-second interval is treated as one evaluation unit")
        print("  - Snippet is positive if ANY second within it has a positive label (>1)")
        print("  - System prediction for snippet uses maximum prediction within the interval")
        
        if self.video_time_periods:
            print("  - FILTERING ENABLED: Only timestamps within recorded video periods are evaluated")
        else:
            print("  - No video time period filtering applied")
        
        self.identity_results = {}
        self.summary_results = {}
        
        for split in ['train', 'val', 'test']:
            split_data = self.matched_pairs_by_split[split]
            
            if not split_data:
                print(f"  {split}: No data to evaluate")
                continue
            
            print(f"  Evaluating {split} split ({len(split_data)} identities)...")
            
            # Evaluate each identity
            identity_metrics = {}
            all_predictions = []
            all_labels = []
            
            for identity, identity_pairs in split_data.items():
                # Evaluate this identity
                metrics = self.evaluate_identity(identity, identity_pairs, split)
                
                if metrics is not None:
                    identity_metrics[identity] = metrics
                    print(f"    {identity}: {metrics['num_samples']} snippets, Accuracy={metrics['best_accuracy']:.3f}, F1={metrics['best_f1']:.3f}, AUC={metrics['auc']:.3f}")
                    
                    # Collect data for macro calculation
                    predictions, labels = self.load_and_align_identity_data(identity_pairs)
                    if predictions is not None:
                        all_predictions.extend(predictions)
                        all_labels.extend(labels)
            
            # Store identity results
            self.identity_results[split] = identity_metrics
            
            # Calculate summary metrics
            if identity_metrics:
                # Micro metrics: average across identities
                micro_metrics = self._calculate_micro_metrics(identity_metrics)
                
                # Macro metrics: all data combined
                macro_metrics = None
                if all_predictions:
                    all_pred_array = np.array(all_predictions)
                    all_label_array = np.array(all_labels)
                    macro_metrics = self.calculate_binary_metrics(all_pred_array, all_label_array)
                    macro_metrics['total_samples'] = len(all_predictions)
                    macro_metrics['total_positives'] = int(all_label_array.sum())
                
                self.summary_results[split] = {
                    'micro_metrics': micro_metrics,
                    'macro_metrics': macro_metrics,
                    'num_identities': len(identity_metrics)
                }
                
                # Print summary
                print(f"    {split} Summary:")
                print(f"      Total snippets: {len(all_predictions)}, Positive: {int(np.array(all_labels).sum())}, Negative: {len(all_predictions) - int(np.array(all_labels).sum())}")
                print(f"      Micro (avg across identities): Accuracy={micro_metrics['best_accuracy']:.3f}, F1={micro_metrics['best_f1']:.3f}, AUC={micro_metrics['auc']:.3f}")
                if macro_metrics:
                    print(f"      Macro (all data combined): Accuracy={macro_metrics['best_accuracy']:.3f}, F1={macro_metrics['best_f1']:.3f}, AUC={macro_metrics['auc']:.3f}")

    def _calculate_micro_metrics(self, identity_metrics):
        """
        Calculate micro metrics by averaging across identities.
        
        Args:
            identity_metrics (dict): Dictionary of {identity: metrics} 
            
        Returns:
            dict: Averaged metrics across identities
        """
        if not identity_metrics:
            return {}
        
        # List of metric names to average
        metric_names = ['auc', 'average_precision', 'best_accuracy', 'best_precision', 
                       'best_recall', 'best_f1', 'positive_rate']
        
        # List of metric names to sum (confusion matrix components)
        sum_metric_names = ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']
        
        micro_metrics = {}
        
        for metric_name in metric_names:
            values = [metrics[metric_name] for metrics in identity_metrics.values() 
                     if metric_name in metrics]
            if values:
                micro_metrics[metric_name] = np.mean(values)
                micro_metrics[f'{metric_name}_std'] = np.std(values)
        
        # Sum up confusion matrix components
        for metric_name in sum_metric_names:
            values = [metrics[metric_name] for metrics in identity_metrics.values() 
                     if metric_name in metrics]
            if values:
                micro_metrics[metric_name] = sum(values)
        
        # Sum up totals
        micro_metrics['total_samples'] = sum(m['num_samples'] for m in identity_metrics.values())
        micro_metrics['total_positives'] = sum(m['num_positives'] for m in identity_metrics.values())
        micro_metrics['num_identities'] = len(identity_metrics)
        
        return micro_metrics

    def save_results(self):
        """
        Save evaluation results to CSV and JSON files.
        """
        print("Saving evaluation results...")
        
        # Save detailed results per identity
        for split in ['train', 'val', 'test']:
            if split in self.identity_results and self.identity_results[split]:
                # Convert to DataFrame for easy CSV saving
                results_list = []
                for identity, metrics in self.identity_results[split].items():
                    result_row = {'identity': identity, 'split': split}
                    result_row.update(metrics)
                    results_list.append(result_row)
                
                df = pd.DataFrame(results_list)
                output_file = os.path.join(self.output_dir, f'identity_results_{split}.csv')
                df.to_csv(output_file, index=False)
                print(f"  Saved {split} identity results to {output_file}")
        
        # Save summary results
        summary_output = os.path.join(self.output_dir, 'summary_results.json')
        with open(summary_output, 'w') as f:
            json.dump(self.summary_results, f, indent=2)
        print(f"  Saved summary results to {summary_output}")
        
        # Save summary as CSV for easy viewing
        summary_rows = []
        for split, split_data in self.summary_results.items():
            # Micro metrics row
            if 'micro_metrics' in split_data and split_data['micro_metrics']:
                micro_row = {'split': split, 'metric_type': 'micro'}
                micro_row.update(split_data['micro_metrics'])
                summary_rows.append(micro_row)
            
            # Macro metrics row
            if 'macro_metrics' in split_data and split_data['macro_metrics']:
                macro_row = {'split': split, 'metric_type': 'macro'}
                macro_row.update(split_data['macro_metrics'])
                summary_rows.append(macro_row)
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_csv = os.path.join(self.output_dir, 'summary_results.csv')
            summary_df.to_csv(summary_csv, index=False)
            print(f"  Saved summary CSV to {summary_csv}")
        
        # Save skipped evaluations information
        if hasattr(self, 'skipped_evaluations') and self.skipped_evaluations['skipped_entries']:
            skipped_output = os.path.join(self.output_dir, 'skipped_evaluations.json')
            with open(skipped_output, 'w') as f:
                json.dump(self.skipped_evaluations, f, indent=2)
            print(f"  Saved skipped evaluations info to {skipped_output}")
            
            # Also save as CSV for easier analysis
            skipped_csv_output = os.path.join(self.output_dir, 'skipped_evaluations.csv')
            skipped_rows = []
            for entry in self.skipped_evaluations['skipped_entries']:
                skipped_rows.append({
                    'person_id': entry['person_id'],
                    'recording_session': entry['recording_session'],
                    'split': entry['split'],
                    'missing_files': ', '.join(entry['expected_files'])
                })
            if skipped_rows:
                df_skipped = pd.DataFrame(skipped_rows)
                df_skipped.to_csv(skipped_csv_output, index=False)
                print(f"  Saved skipped evaluations to {skipped_csv_output}")

    def print_final_summary(self):
        """
        Print a comprehensive final summary of all evaluation results.
        """
        print("\n" + "="*80)
        print("FINAL EVALUATION SUMMARY")
        print("="*80)
        
        for split in ['train', 'val', 'test']:
            if split in self.summary_results and self.summary_results[split]:
                split_data = self.summary_results[split]
                print(f"\n{split.upper()} SPLIT:")
                print(f"  Number of identities: {split_data.get('num_identities', 0)}")
                
                # Macro metrics (all data combined) - calculate this first to get global threshold
                global_threshold = 0.5  # default
                if 'macro_metrics' in split_data and split_data['macro_metrics']:
                    macro = split_data['macro_metrics']
                    global_threshold = macro.get('best_threshold', 0.5)
                    total_samples = macro.get('total_samples', 0)
                    total_positives = macro.get('total_positives', 0)
                    total_negatives = total_samples - total_positives
                    print(f"  MACRO METRICS (all data combined using global threshold {global_threshold}):")
                    print(f"    Snippet Distribution: {total_positives} positive, {total_negatives} negative (total: {total_samples})")
                    print(f"    Accuracy: {macro.get('best_accuracy', 0):.3f}")
                    print(f"    AUC: {macro.get('auc', 0):.3f}")
                    print(f"    F1:  {macro.get('best_f1', 0):.3f}")
                    print(f"    Precision: {macro.get('best_precision', 0):.3f}")
                    print(f"    Recall: {macro.get('best_recall', 0):.3f}")
                    print(f"    Total Snippets: {macro.get('total_samples', 0)}")
                    print(f"    Positive Rate: {macro.get('positive_rate', 0):.3f}")
                    print(f"    Confusion Matrix: TP={macro.get('true_positives', 0)}, TN={macro.get('true_negatives', 0)}, FP={macro.get('false_positives', 0)}, FN={macro.get('false_negatives', 0)}")
                    self._print_confusion_matrix_formatted(macro, "    ")
                
                # Micro metrics (average across identities using individual optimal thresholds)
                if 'micro_metrics' in split_data and split_data['micro_metrics']:
                    micro = split_data['micro_metrics']
                    print(f"  MICRO METRICS (averaged across identities using individual optimal thresholds):")
                    print(f"    Accuracy: {micro.get('best_accuracy', 0):.3f} ± {micro.get('best_accuracy_std', 0):.3f}")
                    print(f"    AUC: {micro.get('auc', 0):.3f} ± {micro.get('auc_std', 0):.3f}")
                    print(f"    F1:  {micro.get('best_f1', 0):.3f} ± {micro.get('best_f1_std', 0):.3f}")
                    print(f"    Precision: {micro.get('best_precision', 0):.3f} ± {micro.get('best_precision_std', 0):.3f}")
                    print(f"    Recall: {micro.get('best_recall', 0):.3f} ± {micro.get('best_recall_std', 0):.3f}")
                    print(f"    Total Snippets: {micro.get('total_samples', 0)}")
                    
                    # Show summed confusion matrix from individual optimal thresholds (will differ from macro)
                    tp_individual = micro.get('true_positives', 0)
                    tn_individual = micro.get('true_negatives', 0)
                    fp_individual = micro.get('false_positives', 0)
                    fn_individual = micro.get('false_negatives', 0)
                    
                    print(f"    Summed CM from individual thresholds: TP={tp_individual}, TN={tn_individual}, FP={fp_individual}, FN={fn_individual}")
                    
                    # Calculate what the metrics would be using the global threshold on each identity
                    # print(f"  ALTERNATIVE: What if all identities used global threshold {global_threshold}:")
                    # self._calculate_and_print_global_threshold_metrics(split, global_threshold)

    def _print_confusion_matrix_formatted(self, metrics, indent=""):
        """
        Print a nicely formatted confusion matrix.
        
        Args:
            metrics (dict): Dictionary containing confusion matrix components
            indent (str): Indentation string for proper formatting
        """
        tp = metrics.get('true_positives', 0)
        tn = metrics.get('true_negatives', 0)
        fp = metrics.get('false_positives', 0)
        fn = metrics.get('false_negatives', 0)
        
        print(f"{indent}    Confusion Matrix:")
        print(f"{indent}    ┌──────────────┬──────────────┐")
        print(f"{indent}    │              │   Predicted  │")
        print(f"{indent}    │              ├──────┬───────┤")
        print(f"{indent}    │              │   0  │   1   │")
        print(f"{indent}    ├──────────────┼──────┼───────┤")
        print(f"{indent}    │ Actual    0  │ {tn:4d} │ {fp:4d}  │")
        print(f"{indent}    │           1  │ {fn:4d} │ {tp:4d}  │")
        print(f"{indent}    └──────────────┴──────┴───────┘")

    def _calculate_and_print_global_threshold_metrics(self, split, global_threshold):
        """
        Calculate and print metrics if all identities used the same global threshold.
        
        Args:
            split (str): Data split name ('train', 'val', 'test')
            global_threshold (float): Global threshold to apply to all identities
        """
        split_data = self.matched_pairs_by_split[split]
        
        total_tp = total_tn = total_fp = total_fn = 0
        
        for identity, identity_pairs in split_data.items():
            # Load data for this identity
            predictions, labels = self.load_and_align_identity_data(identity_pairs)
            
            if predictions is None:
                continue
            
            # Apply global threshold
            binary_preds = (predictions >= global_threshold).astype(int)
            
            # Calculate confusion matrix for this identity
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(labels, binary_preds)
            
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            elif cm.shape == (1, 1):
                if labels[0] == 0:  # All negative labels
                    tn = cm[0, 0]
                    fp = 0
                    fn = 0 
                    tp = 0
                else:  # All positive labels
                    tn = 0
                    fp = 0
                    fn = 0
                    tp = cm[0, 0]
            else:
                tn = fp = fn = tp = 0
            
            # Sum up
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
        
        # Calculate metrics from summed confusion matrix
        if (total_tp + total_fn) > 0:
            recall = total_tp / (total_tp + total_fn)
        else:
            recall = 0.0
        
        if (total_tp + total_fp) > 0:
            precision = total_tp / (total_tp + total_fp)
        else:
            precision = 0.0
            
        if (total_tp + total_tn + total_fp + total_fn) > 0:
            accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        else:
            accuracy = 0.0
            
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        print(f"    Summed CM using global threshold {global_threshold}: TP={total_tp}, TN={total_tn}, FP={total_fp}, FN={total_fn}")
        print(f"    Metrics from global threshold: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        print(f"    (These should match the MACRO metrics above)")

    def test_file_loading_by_split(self, max_identities_per_split=1):
        """
        Test file loading for a few identities in each split to verify data structure.
        
        Args:
            max_identities_per_split (int): Maximum number of identities to test per split
        """
        print("Testing file loading...")
        
        for split_name in ['train', 'val', 'test']:
            split_data = self.matched_pairs_by_split[split_name]
            
            if not split_data:
                continue
                
            tested_count = 0
            for person_id in sorted(split_data.keys()):
                if tested_count >= max_identities_per_split:
                    break
                    
                pairs = split_data[person_id]
                
                for i, (system_file, human_file, session) in enumerate(pairs):
                    # Test system file loading
                    sys_timestamps, sys_predictions = self.smart_load_system_file(system_file)
                    
                    # Test human file loading
                    human_timestamps, human_labels = self.smart_load_human_file(human_file)
                
                tested_count += 1
        
        print("File loading test completed")

    def print_summary(self):
        """
        Print a summary of discovered files and matches.
        """
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        
        print(f"Human evaluation identities: {len(self.human_files_by_identity)}")
        print(f"System output identities: {len(self.system_files_by_identity)}")
        print(f"Identities with matches: {len(self.matched_pairs_by_identity)}")
        
        total_human_files = sum(len(files) for files in self.human_files_by_identity.values())
        total_system_files = sum(len(files) for files in self.system_files_by_identity.values())
        total_matched_pairs = sum(len(pairs) for pairs in self.matched_pairs_by_identity.values())
        
        print(f"Total human files: {total_human_files}")
        print(f"Total system files: {total_system_files}")
        print(f"Total matched pairs: {total_matched_pairs}")
        
        print("\nMatched identities:")
        for person_id in sorted(self.matched_pairs_by_identity.keys()):
            pair_count = len(self.matched_pairs_by_identity[person_id])
            print(f"  {person_id}: {pair_count} pairs")
        
        # TODO: Add evaluation metrics calculation here
        print("\nEvaluation metrics will be implemented in the next step.")

    def run_full_evaluation(self):
        """
        Run the complete evaluation workflow from file discovery to results saving.
        """
        print("Starting complete human vs system evaluation...")
        
        # Step 1: Discover files
        self.discover_human_evaluation_files()
        
        # Step 2: Organize by splits and find matches
        self.organize_files_by_splits()
        
        # Find matching pairs and capture skipped evaluations
        total_matched, total_unmatched, skipped_entries = self.find_matching_pairs()
        
        # Store skipped entries for reporting
        self.skipped_evaluations = {
            'total_matched': total_matched,
            'total_unmatched': total_unmatched,
            'skipped_entries': skipped_entries
        }
        
        # Step 3: Evaluate all identities
        self.evaluate_all_identities()
        
        # Step 4: Save results
        self.save_results()
        
        # Step 5: Print final summary
        self.print_final_summary()
        
        print(f"\nEvaluation complete! Results saved to: {self.output_dir}")
        
        # Print final summary of skipped evaluations
        if skipped_entries:
            print(f"\n" + "="*60)
            print("SKIPPED EVALUATIONS SUMMARY")
            print("="*60)
            print(f"Total evaluations processed: {total_matched + total_unmatched}")
            print(f"✓ Successfully matched: {total_matched}")
            print(f"✗ Skipped (missing CSV files): {total_unmatched}")
            print(f"Skipped evaluation rate: {total_unmatched/(total_matched + total_unmatched)*100:.1f}%")


def main():
    """
    Main function to run human vs system evaluation.
    """
    # Configuration
    system_output_dir = "path_to_your_analysis_root/SNH/alarm_result_output_v9"
    human_evaluation_dir = "path_to_your_analysis_root/SNH/human_evaluation_unified"
    output_dir = "./human_vs_system_evaluation_results"
    data_splits_path = "/Users/kevynzhang/codespace/SNH/data_splits.json"
    video_time_periods_path = "/Users/kevynzhang/codespace/SNH/temp_toolkit/video_time_periods.csv"
    
    # Custom camera to person mapping (adjust as needed based on your data)
    # Default mapping: cam_10 -> p_1, cam_11 -> p_2, cam_12 -> p_3, cam_13 -> p_4, etc.
    camera_to_person_mapping = {
        f'cam_{i}': f'p_{i-9}' for i in range(10, 25)
    }
    
    # Create evaluator and run complete evaluation
    evaluator = HumanSystemEvaluator(
        system_output_dir=system_output_dir,
        human_evaluation_dir=human_evaluation_dir,
        output_dir=output_dir,
        data_splits_path=data_splits_path,
        video_time_periods_path=video_time_periods_path,
        camera_to_person_mapping=camera_to_person_mapping
    )
    
    # Run complete evaluation workflow
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
