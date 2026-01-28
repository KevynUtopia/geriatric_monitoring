import os
import glob
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from .metrics import calculate_f2_score, calculate_fbeta_score, calculate_f5_score, calculate_f10_score, calculate_binary_metrics
from .plots import plot_confusion_matrices, plot_recall_by_identity, plot_f2_score_analysis
from .report import generate_detailed_report
from .results import write_evaluation_results, print_final_summary_report

class FilteredHumanSystemEvaluator:
    def __init__(self, system_output_dir, human_evaluation_filtered_dir, output_dir, data_splits_path, valid_time_interval_path, date_filter=None):
        self.system_output_dir = system_output_dir
        self.human_evaluation_filtered_dir = human_evaluation_filtered_dir
        self.output_dir = output_dir
        self.data_splits_path = data_splits_path
        self.valid_time_interval_path = valid_time_interval_path
        self.valid_time_intervals = self._load_valid_time_intervals()
        os.makedirs(self.output_dir, exist_ok=True)
        self.data_splits = self._load_data_splits()
        self.human_files_by_identity = defaultdict(list)
        self.matched_pairs_by_identity = defaultdict(list)
        self.matched_pairs_by_split = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': defaultdict(list)
        }
        self.identity_results = {}
        self.summary_results = {}
        
        # Set date filter
        self.date_filter = date_filter  # e.g., "06_26" for June 26th

        self.model_type = "timemixer" # "timesnet"
        
        # Set random seed for reproducible bootstrapping
        np.random.seed(42)


    def set_date_filter(self, date_filter=None):
        """
        Set date filter for evaluation.
        
        Args:
            date_filter: Date filter in format "MM_DD" (e.g., "06_26" for June 26th)
        """
        self.date_filter = date_filter
        if date_filter:
            print(f"  📅 DATE FILTER: {date_filter} (only files from 2019_{date_filter})")
        else:
            print(f"  📅 DATE FILTER: None (all dates included)")

    def set_random_seed(self, seed=42):
        """
        Set random seed for reproducible bootstrapping results.
        
        Args:
            seed: Random seed value
        """
        np.random.seed(seed)
        print(f"Random seed set to {seed} for reproducible bootstrapping")

    def _extract_date_from_filename(self, filename):
        """
        Extract date from filename like "ts_recording_2019_06_26_8_15_am_p_22_timemixer.csv"
        Returns date in format "06_26" or None if no date found.
        
        Args:
            filename: Filename to extract date from
            
        Returns:
            str: Date in format "MM_DD" or None if no date found
        """
        import re
        # Pattern to match date in filename: 2019_MM_DD
        pattern = r'2019_(\d{2}_\d{2})'
        match = re.search(pattern, filename)
        if match:
            return match.group(1)  # Returns "06_26" format
        return None

    def _matches_date_filter(self, filename):
        """
        Check if filename matches the date filter.
        
        Args:
            filename: Filename to check
            
        Returns:
            bool: True if filename matches date filter or no filter is set
        """
        if self.date_filter is None:
            return True  # No date filter, include all files
        
        extracted_date = self._extract_date_from_filename(filename)
        if extracted_date is None:
            return False  # No date found in filename, exclude
        
        return extracted_date == self.date_filter

    def _matches_date_filter_from_path(self, file_path):
        """
        Check if file path matches the date filter by extracting date from folder name.
        
        Args:
            file_path: Full file path to check
            
        Returns:
            bool: True if file path matches date filter or no filter is set
        """
        if self.date_filter is None:
            return True  # No date filter, include all files
        
        # Extract date from the folder name (recording session)
        rel_path = os.path.relpath(file_path, self.human_evaluation_filtered_dir)
        path_parts = rel_path.replace('\\', '/').split('/')
        
        if len(path_parts) >= 2:
            recording_session = path_parts[-2]  # e.g., "recording_2019_06_26_8_15_am"
            extracted_date = self._extract_date_from_filename(recording_session)
            if extracted_date is not None:
                return extracted_date == self.date_filter
        
        return False  # No date found in path, exclude

    def get_available_identities(self):
        """
        Get all available identities across all splits for reference.
        
        Returns:
            dict: Dictionary with split names as keys and sets of identity IDs as values
        """
        available = {}
        for split in ['train', 'val', 'test']:
            available[split] = set(self.matched_pairs_by_split[split].keys())
        return available

    def print_available_identities(self):
        """
        Print all available identities for reference when setting filters.
        """
        available = self.get_available_identities()
        print("\n📋 Available identities by split:")
        for split in ['train', 'val', 'test']:
            identities = sorted(available[split])
            print(f"  {split.upper()}: {len(identities)} identities")
            if identities:
                print(f"    {', '.join(identities)}")
        print()

    def _load_data_splits(self):
        try:
            with open(self.data_splits_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading data splits: {e}")
            return {}

    def _load_valid_time_intervals(self):
        """
        Load valid time intervals from CSV file.
        
        Returns:
            dict: Dictionary with recording session as key and list of valid intervals as value
        """
        if not os.path.exists(self.valid_time_interval_path):
            print(f"Warning: Valid time interval file not found: {self.valid_time_interval_path}")
            return {}
        
        import pandas as pd
        df = pd.read_csv(self.valid_time_interval_path)
        valid_intervals = {}
        
        for _, row in df.iterrows():
            recording_session = row.get('Recording Day', '')
            start_time = row.get('Actual Start (seconds)', 0)
            end_time = row.get('Actual End (seconds)', 0)
            
            if recording_session and start_time < end_time:
                if recording_session not in valid_intervals:
                    valid_intervals[recording_session] = []
                valid_intervals[recording_session].append((start_time, end_time))
        print(f"Loaded valid time intervals for {len(valid_intervals)} recording sessions")
        return valid_intervals
            

    def _get_split_for_file(self, recording_session, person_id):
        if 'persons' in self.data_splits:
            person_data = self.data_splits['persons'].get(person_id)
            if person_data:
                for split_name in ['train', 'val', 'test']:
                    if split_name in person_data:
                        split_files = person_data[split_name].get('files', [])
                        for file_info in split_files:
                            if isinstance(file_info, dict) and 'filename' in file_info:
                                # More precise matching: check if the exact recording session matches
                                filename = file_info['filename']
                                expected_filename = f"ts_{recording_session}_{person_id}.csv"
                                if filename == expected_filename:
                                    return split_name
        return None

    def discover_human_evaluation_files(self):
        print("Discovering filtered human evaluation files...")
        pattern = os.path.join(self.human_evaluation_filtered_dir, "*", "*.csv")
        human_files = glob.glob(pattern)

        print(f"Found {len(human_files)} filtered human evaluation files")

        # Apply date filtering if specified
        if self.date_filter:
            print(f"  🔍 Applying date filter: {self.date_filter}")
            original_count = len(human_files)
            human_files = [f for f in human_files if self._matches_date_filter_from_path(f)]
            filtered_count = len(human_files)
            excluded_count = original_count - filtered_count
            print(f"    Date filtering: {original_count} -> {filtered_count} files (excluded {excluded_count})")

        for file_path in human_files:
            rel_path = os.path.relpath(file_path, self.human_evaluation_filtered_dir)
            path_parts = rel_path.replace('\\', '/').split('/')
            
            recording_session = path_parts[-2]
            person_filename = path_parts[-1]
            person_id = person_filename.replace('.csv', '')

            expected_system_filename = f"ts_{recording_session}_{person_id}.csv"
            expected_system_path = os.path.join(self.system_output_dir, expected_system_filename)
            expected_system_filename_tm = f"ts_{recording_session}_{person_id}_{self.model_type}.csv"
            expected_system_path_tm = os.path.join(self.system_output_dir, expected_system_filename_tm)

            # Check if system file exists and matches date filter
            system_file_exists = False
            if os.path.exists(expected_system_path) and self._matches_date_filter(expected_system_filename):
                system_file_exists = True
            elif os.path.exists(expected_system_path_tm) and self._matches_date_filter(expected_system_filename_tm):
                system_file_exists = True

            if system_file_exists:
                self.human_files_by_identity[person_id].append((recording_session, file_path))
                print(f"  Found system file: {expected_system_filename} or {expected_system_filename_tm}")
        
        for person_id in self.human_files_by_identity:
            self.human_files_by_identity[person_id].sort()
        print(f"Organized by {len(self.human_files_by_identity)} identities")

    def organize_files_by_splits(self):
        print("Organizing files by train/val/test splits...")
        split_counts = {'train': 0, 'val': 0, 'test': 0, 'unknown': 0}
        skipped_no_csv = 0
        skipped_date_filter = 0
        
        for person_id, session_files in self.human_files_by_identity.items():
            for recording_session, file_path in session_files:
                split = self._get_split_for_file(recording_session, person_id)
                
                # Check if corresponding CSV file exists in system_output_dir
                expected_system_filename = f"ts_{recording_session}_{person_id}.csv"
                expected_system_path = os.path.join(self.system_output_dir, expected_system_filename)
                expected_system_filename_tm = f"ts_{recording_session}_{person_id}_{self.model_type}.csv"
                expected_system_path_tm = os.path.join(self.system_output_dir, expected_system_filename_tm)
                
                # Check if system file exists and matches date filter
                csv_exists = False
                if os.path.exists(expected_system_path) and self._matches_date_filter(expected_system_filename):
                    csv_exists = True
                elif os.path.exists(expected_system_path_tm) and self._matches_date_filter(expected_system_filename_tm):
                    csv_exists = True
                
                if split in ['train', 'val', 'test'] and csv_exists:
                    self.matched_pairs_by_split[split][person_id].append((None, file_path, recording_session))
                    split_counts[split] += 1
                elif split in ['train', 'val', 'test'] and not csv_exists:
                    if self.date_filter and not (self._matches_date_filter(expected_system_filename) or self._matches_date_filter(expected_system_filename_tm)):
                        print(f"    ⚠️  Skipping {person_id} in {recording_session} ({split} split) - doesn't match date filter {self.date_filter}")
                        skipped_date_filter += 1
                    else:
                        print(f"    ⚠️  Skipping {person_id} in {recording_session} ({split} split) - CSV file not found")
                        skipped_no_csv += 1
                else:
                    split_counts['unknown'] += 1
        
        print(f"Split distribution: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}")
        if split_counts['unknown'] > 0:
            print(f"  Warning: {split_counts['unknown']} files with unknown split")
        if skipped_no_csv > 0:
            print(f"  Skipped: {skipped_no_csv} files due to missing CSV files in system_output_dir")
        if skipped_date_filter > 0:
            print(f"  Skipped: {skipped_date_filter} files due to date filter {self.date_filter}")

    def find_matching_pairs(self):
        """
        Find corresponding system output files for each human evaluation file.
        
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
                    expected_system_filename = f"ts_{recording_session}_{person_id}.csv"
                    expected_system_path = os.path.join(self.system_output_dir, expected_system_filename)
                    expected_system_filename_tm = f"ts_{recording_session}_{person_id}_{self.model_type}.csv"
                    expected_system_path_tm = os.path.join(self.system_output_dir, expected_system_filename_tm)
                    
                    # Check if either system file exists and matches date filter
                    if os.path.exists(expected_system_path) and self._matches_date_filter(expected_system_filename):
                        matched_entries.append((expected_system_path, human_file, recording_session))
                        total_matched += 1
                        print(f"    ✓ Found system file: {expected_system_filename}")
                    elif os.path.exists(expected_system_path_tm) and self._matches_date_filter(expected_system_filename_tm):
                        matched_entries.append((expected_system_path_tm, human_file, recording_session))
                        total_matched += 1
                        print(f"    ✓ Found system file: {expected_system_filename_tm}")
                    else:
                        # Skip this evaluation - CSV file doesn't exist or doesn't match date filter
                        reason = "doesn't match date filter" if self.date_filter and not (self._matches_date_filter(expected_system_filename) or self._matches_date_filter(expected_system_filename_tm)) else "not found"
                        skipped_entry = {
                            'person_id': person_id,
                            'recording_session': recording_session,
                            'expected_files': [expected_system_filename, expected_system_filename_tm],
                            'split': split_name,
                            'reason': reason
                        }
                        skipped_entries.append(skipped_entry)
                        total_unmatched += 1
                        print(f"    ✗ SKIPPING: System file {reason}: {expected_system_filename} or {expected_system_filename_tm}")
                
                if matched_entries:
                    split_data[person_id] = matched_entries
                    self.matched_pairs_by_identity[person_id].extend(matched_entries)
                else:
                    # Remove person from split if no matches found
                    del split_data[person_id]
                    print(f"    ⚠️  Removed {person_id} from {split_name} split - no matching system files")
        
        print(f"\nFile matching summary:")
        print(f"  ✓ Matched pairs: {total_matched}")
        print(f"  ✗ Skipped evaluations: {total_unmatched}")
        
        
        return total_matched, total_unmatched, skipped_entries

    def smart_load_system_file(self, file_path):
        try:
            separate_file_path = file_path.replace('.csv', '_separate.csv')
            separate_file_path_tm = file_path.replace('.csv', f'_{self.model_type}_separate.csv')
            if os.path.exists(separate_file_path):
                df = pd.read_csv(separate_file_path)
                time_col = None
                for col in ['seconds', 'second', 'time', 'timestamp', 'Time', 'Seconds']:
                    if col in df.columns:
                        time_col = col
                        break
                if time_col is None:
                    print(f"    Warning: No time column found in {os.path.basename(separate_file_path)}")
                    return None, None
                if 'anomaly_scores' in df.columns and 'variance_curve' in df.columns:
                    df = df.dropna(subset=[time_col, 'anomaly_scores', 'variance_curve'])
                    if df[time_col].dtype == 'object':
                        try:
                            time_series = pd.to_datetime(df[time_col], format='%H:%M:%S')
                            df[time_col] = time_series.dt.hour * 3600 + time_series.dt.minute * 60 + time_series.dt.second
                        except:
                            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
                    df = df.dropna(subset=[time_col, 'anomaly_scores', 'variance_curve'])
                    if len(df) == 0:
                        return None, None
                    final_predictions = np.clip(df['anomaly_scores'].values + df['variance_curve'].values, 0, 1)
                    print(f"    Loaded separate components from {os.path.basename(separate_file_path)}")
                    return df[time_col].values, final_predictions
                else:
                    print(f"    Warning: Required columns 'anomaly_scores' and 'variance_curve' not found in {os.path.basename(separate_file_path)}")
            elif os.path.exists(separate_file_path_tm):
                df = pd.read_csv(separate_file_path_tm)
                time_col = None
                for col in ['seconds', 'second', 'time', 'timestamp', 'Time', 'Seconds']:
                    if col in df.columns:
                        time_col = col
                        break
                if time_col is None:
                    print(f"    Warning: No time column found in {os.path.basename(separate_file_path_tm)}")
                    return None, None
                if 'anomaly_scores' in df.columns and 'variance_curve' in df.columns:
                    df = df.dropna(subset=[time_col, 'anomaly_scores', 'variance_curve'])
                    if df[time_col].dtype == 'object':
                        try:
                            time_series = pd.to_datetime(df[time_col], format='%H:%M:%S')
                            df[time_col] = time_series.dt.hour * 3600 + time_series.dt.minute * 60 + time_series.dt.second
                        except:
                            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
                    df = df.dropna(subset=[time_col, 'anomaly_scores', 'variance_curve'])
                    if len(df) == 0:
                        return None, None
                    final_predictions = np.maximum(df['anomaly_scores'].values, df['variance_curve'].values)
                    print(f"    Loaded separate components from {os.path.basename(separate_file_path_tm)}")
                    return df[time_col].values, final_predictions
                else:
                    print(f"    Warning: Required columns 'anomaly_scores' and 'variance_curve' not found in {os.path.basename(separate_file_path_tm)}")
            df = pd.read_csv(file_path)
            time_col = None
            for col in ['seconds', 'second', 'time', 'timestamp', 'Time', 'Seconds']:
                if col in df.columns:
                    time_col = col
                    break
            if time_col is None:
                print(f"    Warning: No time column found in {os.path.basename(file_path)}")
                return None, None
            pred_col = None
            for col in ['alarm', 'prediction', 'pred', 'score', 'anomaly', 'alert', 'Alarm', 'Prediction']:
                if col in df.columns:
                    pred_col = col
                    break
            if pred_col is None:
                print(f"    Warning: No prediction column found in {os.path.basename(file_path)}")
                return None, None
            df = df.dropna(subset=[time_col, pred_col])
            if df[time_col].dtype == 'object':
                try:
                    time_series = pd.to_datetime(df[time_col], format='%H:%M:%S')
                    df[time_col] = time_series.dt.hour * 3600 + time_series.dt.minute * 60 + time_series.dt.second
                except:
                    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
            df = df.dropna(subset=[time_col, pred_col])
            if len(df) == 0:
                return None, None
            print(f"    Loaded combined results from {os.path.basename(file_path)} (fallback method)")
            return df[time_col].values, df[pred_col].values
        except Exception as e:
            print(f"    Error loading {os.path.basename(file_path)}: {str(e)}")
            return None, None

    def smart_load_human_file(self, file_path):
        try:
            df = pd.read_csv(file_path)
            time_col = None
            for col in ['second', 'seconds', 'time', 'timestamp', 'Time', 'Second']:
                if col in df.columns:
                    time_col = col
                    break
            if time_col is None:
                print(f"    Warning: No time column found in {os.path.basename(file_path)}")
                return None, None, None
            result_col = None
            for col in ['results', 'result', 'label', 'ground_truth', 'gt', 'annotation', 'Results']:
                if col in df.columns:
                    result_col = col
                    break
            if result_col is None:
                print(f"    Warning: No result column found in {os.path.basename(file_path)}")
                return None, None, None
            count_col = None
            for col in ['count', 'Count', 'counts', 'Counts']:
                if col in df.columns:
                    count_col = col
                    break
            if count_col is None:
                print(f"    Warning: No count column found in {os.path.basename(file_path)}")
                return None, None, None
            df = df.dropna(subset=[time_col, result_col, count_col])
            if df[time_col].dtype == 'object':
                try:
                    time_series = pd.to_datetime(df[time_col], format='%H:%M:%S')
                    df[time_col] = time_series.dt.hour * 3600 + time_series.dt.minute * 60 + time_series.dt.second
                except:
                    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
            df = df.dropna(subset=[time_col, result_col, count_col])
            if len(df) == 0:
                return None, None, None
            print(f"    Loaded human evaluation from {os.path.basename(file_path)}")
            return df[time_col].values, df[result_col].values, df[count_col].values
        except Exception as e:
            print(f"    Error loading {os.path.basename(file_path)}: {str(e)}")
            return None, None, None



    def process_into_snippets(self, timestamps, predictions, labels, counts, snippet_duration=20, valid_intervals=None):
        if len(timestamps) == 0:
            return None, None
        min_time = int(timestamps.min())
        max_time = int(timestamps.max())
        snippet_predictions = []
        snippet_labels = []
        for start_time in range(min_time, max_time + 1, snippet_duration):
            end_time = start_time + snippet_duration
            # If valid_intervals is provided, check if this snippet overlaps any valid interval
            if valid_intervals is not None:
                overlaps = False
                for vstart, vend in valid_intervals:
                    if not (end_time <= vstart or start_time >= vend):
                        overlaps = True
                        break
                if not overlaps:
                    continue  # Skip this snippet
            mask = (timestamps >= start_time) & (timestamps < end_time)
            if not np.any(mask):
                continue
            interval_predictions = predictions[mask]
            interval_labels = labels[mask]
            interval_counts = counts[mask]
            snippet_pred = np.max(interval_predictions)
            snippet_label = 1 if np.any((interval_labels > 0) & (interval_counts > 1)) else 0
            snippet_predictions.append(snippet_pred)
            snippet_labels.append(snippet_label)
        if not snippet_predictions:
            return None, None
        return np.array(snippet_predictions), np.array(snippet_labels)

    def load_and_align_identity_data(self, identity_pairs, return_snippet_count=False):
        all_predictions = []
        all_labels = []
        valid_snippet_count = 0
        for system_file, human_file, session in identity_pairs:
            sys_times, sys_preds = self.smart_load_system_file(system_file)
            human_times, human_labels, human_counts = self.smart_load_human_file(human_file)
            if sys_times is None or human_times is None:
                continue
            sys_dict = dict(zip(sys_times, sys_preds))
            session_predictions = []
            session_labels = []
            session_counts = []
            session_timestamps = []
            for i, timestamp in enumerate(human_times):
                pred = sys_dict.get(timestamp, 0.0)
                session_predictions.append(pred)
                session_labels.append(human_labels[i])
                session_counts.append(human_counts[i])
                session_timestamps.append(timestamp)
            if session_predictions:
                session_pred_array = np.array(session_predictions)
                session_label_array = np.array(session_labels)
                session_count_array = np.array(session_counts)
                session_time_array = np.array(session_timestamps)
                valid_intervals = self.valid_time_intervals.get(session, None)
                if valid_intervals is None:
                    continue
                snippet_preds, snippet_labels = self.process_into_snippets(
                    session_time_array, session_pred_array, session_label_array, session_count_array,
                    valid_intervals=valid_intervals
                )
                if snippet_preds is not None:
                    all_predictions.extend(snippet_preds)
                    all_labels.extend(snippet_labels)
                    valid_snippet_count += len(snippet_preds)
        if not all_predictions:
            if return_snippet_count:
                return None, None, 0
            return None, None
        if return_snippet_count:
            return np.array(all_predictions), np.array(all_labels), valid_snippet_count
        return np.array(all_predictions), np.array(all_labels)

    def bootstrap_metrics(self, identity_metrics, identity_snippet_data, n_bootstrap=100, save_bootstrap_results=True, split_name=None, output_dir=None):
        """
        Calculate metrics using bootstrapping to get more robust std estimates.
        
        For each identity, bootstrap snippets with replacement for n_bootstrap rounds.
        This provides more accurate uncertainty estimates by sampling the actual time-series data.
        
        Args:
            identity_metrics: Dictionary of identity metrics
            identity_snippet_data: Dictionary of raw snippet data per identity
            n_bootstrap: Number of bootstrap rounds
            save_bootstrap_results: Whether to save individual bootstrap results as CSV
            split_name: Name of the split (for saving files)
            output_dir: Directory to save bootstrap results
            
        Returns:
            dict: Dictionary with mean and std for each metric
        """
        if not identity_metrics or not identity_snippet_data:
            return {}
        
        # Get all metric names from the first identity
        first_identity = list(identity_metrics.values())[0]
        metric_names = [key for key in first_identity.keys() 
                       if key not in ['identity', 'split', 'num_samples', 'num_positives']]
        
        bootstrap_results = {metric: [] for metric in metric_names}
        
        # Store individual identity bootstrap results
        identity_bootstrap_results = {identity: {metric: [] for metric in metric_names} 
                                    for identity in identity_metrics.keys()}
        
        # Bootstrap sampling
        for bootstrap_round in range(n_bootstrap):
            # For each identity, bootstrap their snippets
            bootstrapped_identity_metrics = {}
            
            for identity in identity_metrics.keys():
                if identity in identity_snippet_data:
                    snippet_data = identity_snippet_data[identity]
                    predictions = snippet_data['predictions']
                    labels = snippet_data['labels']
                    
                    # Bootstrap snippets within this identity
                    n_snippets = len(predictions)
                    if n_snippets > 0:
                        # Sample snippet indices with replacement
                        bootstrap_indices = np.random.choice(n_snippets, size=n_snippets, replace=True)
                        bootstrapped_predictions = predictions[bootstrap_indices]
                        bootstrapped_labels = labels[bootstrap_indices]
                        
                        # Calculate metrics for this bootstrapped sample
                        bootstrapped_metrics = calculate_binary_metrics(bootstrapped_predictions, bootstrapped_labels)
                        
                        # Calculate optimal threshold metrics for bootstrapped sample
                        optimal_metrics = self.calculate_optimal_threshold_metrics(bootstrapped_predictions, bootstrapped_labels)
                        bootstrapped_metrics.update(optimal_metrics)
                        
                        bootstrapped_identity_metrics[identity] = bootstrapped_metrics
                        
                        # Store individual identity results
                        for metric in metric_names:
                            if metric in bootstrapped_metrics:
                                identity_bootstrap_results[identity][metric].append(bootstrapped_metrics[metric])
            
            # Calculate micro metrics for this bootstrap round
            if bootstrapped_identity_metrics:
                micro_metrics = self._calculate_micro_metrics_simple(bootstrapped_identity_metrics)
                
                # Store results
                for metric in metric_names:
                    if metric in micro_metrics:
                        bootstrap_results[metric].append(micro_metrics[metric])
        
        # Save individual identity bootstrap results as CSV files
        if save_bootstrap_results and split_name and output_dir:
            bootstrap_dir = os.path.join(output_dir, 'bootstrap_results', split_name)
            os.makedirs(bootstrap_dir, exist_ok=True)
            
            for identity in identity_bootstrap_results:
                identity_results = identity_bootstrap_results[identity]
                
                # Create DataFrame with bootstrap results
                bootstrap_data = []
                for round_num in range(n_bootstrap):
                    row = {'bootstrap_round': round_num}
                    for metric in metric_names:
                        if metric in identity_results and len(identity_results[metric]) > round_num:
                            row[metric] = identity_results[metric][round_num]
                        else:
                            row[metric] = None
                    bootstrap_data.append(row)
                
                df = pd.DataFrame(bootstrap_data)
                csv_path = os.path.join(bootstrap_dir, f'{identity}_bootstrap_results.csv')
                df.to_csv(csv_path, index=False)
                print(f"    Saved bootstrap results for {identity} to {csv_path}")
        
        # Calculate mean and std from bootstrap results
        result = {}
        for metric, values in bootstrap_results.items():
            if values:
                result[metric] = np.mean(values)
                result[f'{metric}_std'] = np.std(values)
        
        return result

    def _calculate_micro_metrics_simple(self, identity_metrics):
        """
        Calculate micro metrics without bootstrapping (used internally by bootstrap_metrics).
        """
        average_metric_names = [
            'auc', 'average_precision', 'best_accuracy', 'best_precision', 'best_recall', 
            'best_f1', 'best_f2', 'best_f5', 'best_f10', 'best_cohen_kappa', 'positive_rate',
            'optimal_threshold', 'optimal_f1', 'optimal_f2', 'optimal_f5', 'optimal_f10',
            'optimal_precision', 'optimal_recall', 'optimal_accuracy'
        ]
        sum_metric_names = ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']
        micro_metrics = {}
        
        for metric_name in average_metric_names:
            values = [metrics[metric_name] for metrics in identity_metrics.values() 
                     if metric_name in metrics]
            if values:
                micro_metrics[metric_name] = np.mean(values)
        
        for metric_name in sum_metric_names:
            values = [metrics.get(metric_name, 0) for metrics in identity_metrics.values()]
            micro_metrics[metric_name] = sum(values)
        
        micro_metrics['total_samples'] = sum(m['num_samples'] for m in identity_metrics.values())
        micro_metrics['total_positives'] = sum(m['num_positives'] for m in identity_metrics.values())
        micro_metrics['num_identities'] = len(identity_metrics)
        
        return micro_metrics

    def calculate_micro_metrics(self, identity_metrics, identity_snippet_data=None, split_name=None, output_dir=None):
        # Use bootstrapping for std calculation
        if identity_snippet_data:
            bootstrap_metrics = self.bootstrap_metrics(
                identity_metrics, 
                identity_snippet_data, 
                n_bootstrap=100,
                save_bootstrap_results=True,
                split_name=split_name,
                output_dir=output_dir
            )
        else:
            bootstrap_metrics = {}
        
        # Calculate traditional micro metrics
        average_metric_names = ['auc', 'average_precision', 'best_accuracy', 'best_precision', 'best_recall', 'best_f1', 'best_f2', 'best_f5', 'best_f10', 'best_cohen_kappa', 'positive_rate']
        sum_metric_names = ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']
        micro_metrics = {}
        
        # Use bootstrap results for average metrics if available
        for metric_name in average_metric_names:
            if metric_name in bootstrap_metrics:
                micro_metrics[metric_name] = bootstrap_metrics[metric_name]
                micro_metrics[f'{metric_name}_std'] = bootstrap_metrics.get(f'{metric_name}_std', 0)
            else:
                # Fallback to traditional calculation if no bootstrap data
                values = [metrics[metric_name] for metrics in identity_metrics.values() 
                         if metric_name in metrics]
                if values:
                    micro_metrics[metric_name] = np.mean(values)
                    micro_metrics[f'{metric_name}_std'] = 0  # No std without bootstrapping
        
        # Calculate sum metrics traditionally
        for metric_name in sum_metric_names:
            values = [metrics.get(metric_name, 0) for metrics in identity_metrics.values()]
            micro_metrics[metric_name] = sum(values)
        
        micro_metrics['total_samples'] = sum(m['num_samples'] for m in identity_metrics.values())
        micro_metrics['total_positives'] = sum(m['num_positives'] for m in identity_metrics.values())
        micro_metrics['num_identities'] = len(identity_metrics)
        
        return micro_metrics

    def calculate_macro_metrics(self, identity_metrics):
        tp_sum = sum(m.get('true_positives', 0) for m in identity_metrics.values())
        tn_sum = sum(m.get('true_negatives', 0) for m in identity_metrics.values())
        fp_sum = sum(m.get('false_positives', 0) for m in identity_metrics.values())
        fn_sum = sum(m.get('false_negatives', 0) for m in identity_metrics.values())
        total_samples = tp_sum + tn_sum + fp_sum + fn_sum
        total_positives = tp_sum + fn_sum
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f2 = calculate_f2_score(precision, recall)
        f5 = calculate_f5_score(precision, recall)
        f10 = calculate_f10_score(precision, recall)
        macro_metrics = {
            'auc': np.mean([m['auc'] for m in identity_metrics.values()]),
            'average_precision': np.mean([m['average_precision'] for m in identity_metrics.values()]),
            'best_accuracy': (tp_sum + tn_sum) / total_samples if total_samples > 0 else 0,
            'best_precision': precision,
            'best_recall': recall,
            'best_f1': f1,
            'best_f2': f2,
            'best_f5': f5,
            'best_f10': f10,
            'best_cohen_kappa': np.mean([m.get('best_cohen_kappa', 0) for m in identity_metrics.values()]),
            'best_threshold': 0.5,
            'true_positives': tp_sum,
            'true_negatives': tn_sum,
            'false_positives': fp_sum,
            'false_negatives': fn_sum,
            'num_samples': total_samples,
            'num_positives': total_positives,
            'positive_rate': total_positives / total_samples if total_samples > 0 else 0,
            'total_samples': total_samples,
            'total_positives': total_positives
        }
        return macro_metrics

    def calculate_topN_metrics(self, identity_metrics, N_list=[3]):
        sorted_identities = sorted(identity_metrics.items(), key=lambda x: x[1]['best_f2'], reverse=True)
        metrics_at_N = {}
        metric_names = [
            'auc', 'average_precision', 'best_accuracy', 'best_precision', 'best_recall',
            'best_f1', 'best_f2', 'best_f5', 'best_f10', 'best_cohen_kappa'
        ]
        
        for N in N_list:
            topN = sorted_identities[:N]
            if not topN:
                continue
            
            # Create identity_metrics for top N
            topN_identity_metrics = {k: v for k, v in topN}
            
            # Get snippet data for top N identities
            topN_snippet_data = {}
            if hasattr(self, 'identity_snippet_data'):
                for split in self.identity_snippet_data:
                    split_snippet_data = self.identity_snippet_data[split]
                    for identity in topN_identity_metrics.keys():
                        if identity in split_snippet_data:
                            topN_snippet_data[identity] = split_snippet_data[identity]
            
            # Use bootstrapping for top N metrics
            if topN_snippet_data:
                bootstrap_metrics = self.bootstrap_metrics(
                    topN_identity_metrics, 
                    topN_snippet_data, 
                    n_bootstrap=100,
                    save_bootstrap_results=False  # Don't save for top-N to avoid confusion
                )
            else:
                bootstrap_metrics = {}
            
            metrics = {}
            for metric in metric_names:
                if metric in bootstrap_metrics:
                    metrics[f'{metric}@{N}'] = bootstrap_metrics[metric]
                    metrics[f'{metric}@{N}_std'] = bootstrap_metrics.get(f'{metric}_std', 0)
            
            metrics_at_N[N] = metrics
        
        return metrics_at_N

    def calculate_optimal_threshold_metrics(self, predictions, labels):
        """
        Calculate metrics at optimal threshold for each identity.
        
        Args:
            predictions: Array of prediction scores
            labels: Array of true labels
            
        Returns:
            dict: Dictionary with optimal threshold and corresponding metrics
        """
        if len(np.unique(labels)) == 1:
            # Handle edge case where all labels are the same
            return {
                'optimal_threshold': 0.5,
                'optimal_f1': 0.0,
                'optimal_f2': 0.0,
                'optimal_f5': 0.0,
                'optimal_f10': 0.0,
                'optimal_precision': 0.0,
                'optimal_recall': 0.0,
                'optimal_accuracy': 1.0 if labels[0] == 0 else 1.0
            }
        
        thresholds = np.linspace(0, 1, 101)
        best_accuracy = 0
        optimal_threshold = 0.5
        optimal_metrics = {}
        
        for threshold in thresholds:
            binary_preds = (predictions >= threshold).astype(int)
            
            # Calculate metrics at this threshold
            tp = np.sum((binary_preds == 1) & (labels == 1))
            tn = np.sum((binary_preds == 0) & (labels == 0))
            fp = np.sum((binary_preds == 1) & (labels == 0))
            fn = np.sum((binary_preds == 0) & (labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f2 = calculate_f2_score(precision, recall)
            f5 = calculate_f5_score(precision, recall)
            f10 = calculate_f10_score(precision, recall)
            
            # Use accuracy to determine optimal threshold
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                optimal_threshold = threshold
                optimal_metrics = {
                    'optimal_threshold': optimal_threshold,
                    'optimal_f1': f1,
                    'optimal_f2': f2,
                    'optimal_f5': f5,
                    'optimal_f10': f10,
                    'optimal_precision': precision,
                    'optimal_recall': recall,
                    'optimal_accuracy': accuracy,
                    'optimal_tp': int(tp),
                    'optimal_tn': int(tn),
                    'optimal_fp': int(fp),
                    'optimal_fn': int(fn)
                }
        
        return optimal_metrics

    def evaluate_identity(self, identity, identity_pairs, split):
        predictions, labels = self.load_and_align_identity_data(identity_pairs)
        if predictions is None:
            print(f"    Warning: No valid data for {identity} in {split}")
            return None
        
        # Calculate basic metrics
        num_positive = int(labels.sum())
        num_negative = len(labels) - num_positive
        total_samples = len(labels)
        positive_rate = num_positive / total_samples if total_samples > 0 else 0
        
        print(f"    {identity} snippet distribution: {num_positive} positive, {num_negative} negative (total: {total_samples} snippets, positive rate: {positive_rate:.3f})")
        
        # Calculate standard metrics
        metrics = calculate_binary_metrics(predictions, labels)
        
        # Calculate optimal threshold metrics
        optimal_metrics = self.calculate_optimal_threshold_metrics(predictions, labels)
        
        # Combine all metrics
        metrics.update(optimal_metrics)
        
        return metrics

    def evaluate_all_identities(self):
        print("Calculating evaluation metrics using 5-second snippet-based approach...")
        self.identity_results = {}
        self.summary_results = {}
        self.valid_snippet_counts = {}  # Track valid snippet count per split
        self.identity_snippet_data = {}  # Store raw snippet data for bootstrapping
        
        for split in ['train', 'val', 'test']:
            split_data = self.matched_pairs_by_split[split].copy()
            
            if not split_data:
                print(f"  {split}: No data to evaluate")
                self.valid_snippet_counts[split] = 0
                continue
                
            print(f"  Evaluating {split} split ({len(split_data)} identities)...")
            identity_metrics = {}
            identity_snippet_data = {}
            valid_snippet_count = 0
            for identity in sorted(split_data.keys()):
                identity_pairs = split_data[identity]
                predictions, labels, snippet_count = self.load_and_align_identity_data(identity_pairs, return_snippet_count=True)
                if predictions is not None:
                    # Store raw snippet data for bootstrapping
                    identity_snippet_data[identity] = {
                        'predictions': predictions,
                        'labels': labels
                    }
                    
                    metrics = calculate_binary_metrics(predictions, labels)
                    identity_metrics[identity] = metrics
                    valid_snippet_count += snippet_count

            self.identity_results[split] = identity_metrics
            self.identity_snippet_data[split] = identity_snippet_data
            self.valid_snippet_counts[split] = valid_snippet_count
            
            if identity_metrics:
                # Auto-calculate topN based on actual number of identities
                num_identities = len(identity_metrics)
                if num_identities >= 7:
                    N_list = [3, 5, 7]
                elif num_identities >= 5:
                    N_list = [3, 5]
                elif num_identities >= 3:
                    N_list = [3]
                else:
                    N_list = [num_identities] if num_identities > 0 else []
                
                micro_metrics = self.calculate_micro_metrics(identity_metrics, identity_snippet_data, split_name=split, output_dir=self.output_dir)
                macro_metrics = self.calculate_macro_metrics(identity_metrics)
                topN_metrics = self.calculate_topN_metrics(identity_metrics, N_list=N_list)
                self.summary_results[split] = {
                    'micro_metrics': micro_metrics,
                    'macro_metrics': macro_metrics,
                    'topN_metrics': topN_metrics,
                    'num_identities': len(identity_metrics)
                }
                print(f"  {split}: AUC={micro_metrics['auc']:.4f}, F1={micro_metrics['best_f1']:.4f}, samples={micro_metrics['total_samples']}")
            else:
                print(f"  {split}: No valid data")

    def save_results(self, save_dir=None):
        if save_dir is None:
            save_dir = self.output_dir
        write_evaluation_results(
            identity_results=self.identity_results,
            summary_results=self.summary_results,
            save_dir=save_dir,
            skipped_evaluations=getattr(self, 'skipped_evaluations', None)
        )

    def print_final_summary(self):
        print_final_summary_report(self.summary_results, getattr(self, 'valid_snippet_counts', None))

    def run_full_evaluation(self, save_dir=None):
        print("Starting complete filtered human vs system evaluation...")

        self.discover_human_evaluation_files()
        self.organize_files_by_splits()
        
        # Find matching pairs and capture skipped evaluations
        total_matched, total_unmatched, skipped_entries = self.find_matching_pairs()
        
        # Store skipped entries for reporting
        self.skipped_evaluations = {
            'total_matched': total_matched,
            'total_unmatched': total_unmatched,
            'skipped_entries': skipped_entries
        }
        
        self.evaluate_all_identities()
        self.save_results(save_dir)
        self.print_final_summary()
        final_save_dir = save_dir if save_dir else self.output_dir
        print(f"\nEvaluation complete! Results saved to: {final_save_dir}") 
        
        # Print final summary of skipped evaluations
        if skipped_entries:
            print(f"\n" + "="*60)
            print("SKIPPED EVALUATIONS SUMMARY")
            print("="*60)
            print(f"Total evaluations processed: {total_matched + total_unmatched}")
            print(f"✓ Successfully matched: {total_matched}")
            print(f"✗ Skipped (missing CSV files): {total_unmatched}")
            print(f"Skipped evaluation rate: {total_unmatched/(total_matched + total_unmatched)*100:.1f}%") 