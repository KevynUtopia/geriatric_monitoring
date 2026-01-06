import os
import glob
import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from .metrics import calculate_f2_score, calculate_fbeta_score, calculate_f5_score, calculate_f10_score, calculate_binary_metrics
from .plots import plot_confusion_matrices, plot_recall_by_identity, plot_f2_score_analysis, plot_human_vs_system_timeseries
from .report import generate_detailed_report
from .results import write_evaluation_results, print_final_summary_report

class GCLHumanSystemEvaluator:
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
        self.timestamp_results = {}
        self.summary_results = {}
        
        # Set date filter
        self.date_filter = date_filter  # e.g., "06_26" for June 26th
        
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


        return valid_intervals

    def _extract_date_from_filename(self, filename):
        """
        Extract date from filename like "anomaly_scores_recording_2019_06_26_8_15_am_cam_10.csv"
        Returns date in format "06_26" or None if no date found.
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
        """
        if self.date_filter is None:
            return True  # No date filter, include all files
        
        extracted_date = self._extract_date_from_filename(filename)
        if extracted_date is None:
            return False  # No date found in filename, exclude
        
        return extracted_date == self.date_filter

    def _parse_feature_index_to_timestamp(self, feature_index):
        """
        Parse feature_index like "110316_20" to timestamp in seconds.
        feature_index format: HHMMSS_INDEX where INDEX is 1-second intervals from start time.
        """
        try:
            time_part, index_part = feature_index.split('_')
            hours = int(time_part[:2])
            minutes = int(time_part[2:4])
            seconds = int(time_part[4:6])
            index = int(index_part)
            
            # Convert to total seconds from midnight
            base_seconds = hours * 3600 + minutes * 60 + seconds
            # Add 1-second intervals
            total_seconds = base_seconds + (index * 1)
            return total_seconds
        except:
            return None

    def discover_gcl_system_files(self):
        """Discover GCL system result files."""
        print("Discovering GCL system result files...")
        pattern = os.path.join(self.system_output_dir, "*.csv")
        system_files = glob.glob(pattern)
        
        print(f"Found {len(system_files)} GCL system result files")
        
        # Apply date filtering if specified
        if self.date_filter:
            print(f"  🔍 Applying date filter: {self.date_filter}")
            original_count = len(system_files)
            system_files = [f for f in system_files if self._matches_date_filter(os.path.basename(f))]
            filtered_count = len(system_files)
            excluded_count = original_count - filtered_count
        
        return system_files

    def discover_human_evaluation_files(self):
        """Discover human evaluation files."""
        print("Discovering human evaluation files...")
        pattern = os.path.join(self.human_evaluation_filtered_dir, "*", "*.csv")
        human_files = glob.glob(pattern)
        
        print(f"Found {len(human_files)} human evaluation files")
        
        # Apply date filtering if specified
        if self.date_filter:
            print(f"  🔍 Applying date filter: {self.date_filter}")
            original_count = len(human_files)
            human_files = [f for f in human_files if self._matches_date_filter_from_path(f)]
            filtered_count = len(human_files)
            excluded_count = original_count - filtered_count
        
        return human_files

    def _matches_date_filter_from_path(self, file_path):
        """Check if file path matches the date filter by extracting date from folder name."""
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

    def load_gcl_system_file(self, file_path):
        """Load GCL system result file and convert to timestamp-based format."""
        try:
            df = pd.read_csv(file_path)
            
            # Extract recording session from filename
            filename = os.path.basename(file_path)
            
            # Check if this is an aggregated file (format: aggregated_YYYY_MM_DD.csv)
            if filename.startswith('aggregated_'):
                # For aggregated files, extract date and create recording session
                date_match = re.search(r'aggregated_(\d{4}_\d{2}_\d{2})', filename)
                if date_match:
                    date_str = date_match.group(1)
                    # Create a generic recording session name based on date
                    recording_session = f"recording_{date_str}_aggregated"
                else:
                    recording_session = filename.replace('aggregated_', '').replace('.csv', '')
            else:
                # Original format: anomaly_scores_recording_2019_07_13_6_50_am_cam_10.csv
                recording_session = filename.replace('anomaly_scores_', '').replace('.csv', '')
                # Remove camera part (e.g., _cam_10)
                recording_session = re.sub(r'_cam_\d+$', '', recording_session)
            
            # Parse timestamps and collect data
            data_with_timestamps = []
            
            for _, row in df.iterrows():
                # For aggregated files, timestamp is already parsed and stored in 'timestamp' column
                if 'timestamp' in df.columns:
                    timestamp = row['timestamp']
                else:
                    # For original files, parse from feature_index
                    feature_index = row['feature_index']
                    timestamp = self._parse_feature_index_to_timestamp(feature_index)
                
                if timestamp is not None:
                    # Use anomaly_score_max as the prediction score
                    prediction = row['anomaly_score_max']
                    data_with_timestamps.append((timestamp, prediction))
            
            if data_with_timestamps:
                # Sort by timestamp to ensure chronological order
                data_with_timestamps.sort(key=lambda x: x[0])
                
                # Extract sorted timestamps and predictions
                timestamps = [item[0] for item in data_with_timestamps]
                predictions = [item[1] for item in data_with_timestamps]
                
                
                
                return recording_session, np.array(timestamps), np.array(predictions)
            else:
                print(f"    Warning: No valid timestamps found in {os.path.basename(file_path)}")
                return None, None, None
                
        except Exception as e:
            print(f"    Error loading {os.path.basename(file_path)}: {str(e)}")
            return None, None, None

    def load_human_evaluation_file(self, file_path):
        """Load human evaluation file."""
        try:
            df = pd.read_csv(file_path)
            
            # Find time column
            time_col = None
            for col in ['second', 'seconds', 'time', 'timestamp', 'Time', 'Second']:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col is None:
                print(f"    Warning: No time column found in {os.path.basename(file_path)}")
                return None, None, None
            
            # Find result column
            result_col = None
            for col in ['results', 'result', 'label', 'ground_truth', 'gt', 'annotation', 'Results']:
                if col in df.columns:
                    result_col = col
                    break
            
            if result_col is None:
                print(f"    Warning: No result column found in {os.path.basename(file_path)}")
                return None, None, None
            
            # Find count column
            count_col = None
            for col in ['count', 'Count', 'counts', 'Counts']:
                if col in df.columns:
                    count_col = col
                    break
            
            if count_col is None:
                print(f"    Warning: No count column found in {os.path.basename(file_path)}")
                return None, None, None
            
            # Clean data
            df = df.dropna(subset=[time_col, result_col, count_col])
            
            if len(df) == 0:
                print(f"    Warning: No valid data in {os.path.basename(file_path)}")
                return None, None, None
            
            # Convert time to seconds if needed
            if df[time_col].dtype == 'object':
                try:
                    time_series = pd.to_datetime(df[time_col], format='%H:%M:%S')
                    df[time_col] = time_series.dt.hour * 3600 + time_series.dt.minute * 60 + time_series.dt.second
                except:
                    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
            
            df = df.dropna(subset=[time_col, result_col, count_col])
            
            if len(df) == 0:
                print(f"    Warning: No valid data after conversion in {os.path.basename(file_path)}")
                return None, None, None
            
            return df[time_col].values, df[result_col].values, df[count_col].values
            
        except Exception as e:
            print(f"    Error loading {os.path.basename(file_path)}: {str(e)}")
            return None, None, None

    def aggregate_human_evaluations_by_timestamp(self, human_files):
        """
        Aggregate human evaluation results across all identities for each timestamp.
        A timestamp is considered anomalous if ANY identity is recognized as anomalous.
        """
        
        # Dictionary to store aggregated results: {recording_session: {timestamp: aggregated_label}}
        aggregated_data = defaultdict(lambda: defaultdict(list))
        
        for human_file in human_files:
            # Extract recording session and person ID from file path
            rel_path = os.path.relpath(human_file, self.human_evaluation_filtered_dir)
            path_parts = rel_path.replace('\\', '/').split('/')
            
            if len(path_parts) >= 2:
                recording_session = path_parts[-2]
                person_filename = path_parts[-1]
                person_id = person_filename.replace('.csv', '')
                
                # Load human evaluation data
                human_times, human_labels, human_counts = self.load_human_evaluation_file(human_file)
                
                if human_times is not None:
                    for i, timestamp in enumerate(human_times):
                        # A timestamp is anomalous if label > 0 AND count > 1 (multiple annotators agree)
                        is_anomalous = 1 if (human_labels[i] > 0 and human_counts[i] > 1) else 0
                        aggregated_data[recording_session][timestamp].append(is_anomalous)
        
        # Convert to final format: {recording_session: {timestamp: final_label}}
        final_aggregated = {}
        for recording_session, timestamp_data in aggregated_data.items():
            final_aggregated[recording_session] = {}
            for timestamp, anomaly_votes in timestamp_data.items():
                # If ANY identity voted for anomaly, the timestamp is anomalous
                final_label = 1 if any(anomaly_votes) else 0
                final_aggregated[recording_session][timestamp] = final_label
        
        return final_aggregated

    def match_system_and_human_data(self, system_files, aggregated_human_data):
        """
        Match system predictions with aggregated human labels by timestamp.
        """
        
        matched_data = {}
        
        for system_file in system_files:
            recording_session, sys_timestamps, sys_predictions = self.load_gcl_system_file(system_file)
            
            if recording_session is None:
                continue
            
            # For aggregated files, try to find matching human evaluation session by date
            if recording_session.endswith('_aggregated'):
                # Extract date from aggregated session name
                date_match = re.search(r'recording_(\d{4}_\d{2}_\d{2})_aggregated', recording_session)
                if date_match:
                    date_str = date_match.group(1)
                    # Find matching human evaluation session by date
                    matching_human_session = None
                    for human_session in aggregated_human_data.keys():
                        if date_str in human_session:
                            matching_human_session = human_session
                            break
                    
                    if matching_human_session is None:
                        continue
                    
                    human_timestamp_data = aggregated_human_data[matching_human_session]
                else:
                    continue
            else:
                # Original logic for non-aggregated files
                if recording_session not in aggregated_human_data:
                    continue
                human_timestamp_data = aggregated_human_data[recording_session]
            
            # Find matching timestamps
            matched_timestamps = []
            matched_predictions = []
            matched_labels = []
            
            for i, timestamp in enumerate(sys_timestamps):
                if timestamp in human_timestamp_data:
                    matched_timestamps.append(timestamp)
                    matched_predictions.append(sys_predictions[i])
                    matched_labels.append(human_timestamp_data[timestamp])
            
            if matched_timestamps:
                matched_data[recording_session] = {
                    'timestamps': np.array(matched_timestamps),
                    'predictions': np.array(matched_predictions),
                    'labels': np.array(matched_labels)
                }
        

        return matched_data

    def process_into_snippets(self, timestamps, predictions, labels, snippet_duration=5, valid_intervals=None):
        """
        Process matched data into snippets for evaluation.
        """
        if len(timestamps) == 0:
            return None, None
        
        min_time = int(timestamps.min())
        max_time = int(timestamps.max())
        snippet_predictions = []
        snippet_labels = []
        
        for start_time in range(min_time, max_time + 1, snippet_duration):
            end_time = start_time + snippet_duration
            
            # Check if this snippet overlaps any valid interval
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
            
            # Snippet prediction uses maximum prediction within the interval
            snippet_pred = np.mean(interval_predictions)
            # Snippet is positive if ANY timestamp within it has a positive label
            snippet_label = 1 if np.any(interval_labels > 0) else 0
            
            snippet_predictions.append(snippet_pred)
            snippet_labels.append(snippet_label)
        
        if not snippet_predictions:
            return None, None
        
        return np.array(snippet_predictions), np.array(snippet_labels)

    def load_and_align_timestamp_data(self, matched_data, return_snippet_count=False):
        """
        Load and align all timestamp data, processing into snippets.
        """
        all_predictions = []
        all_labels = []
        valid_snippet_count = 0
        
        for recording_session, data in matched_data.items():
            timestamps = data['timestamps']
            predictions = data['predictions']
            labels = data['labels']
            
            # Get valid time intervals for this recording session
            valid_intervals = self.valid_time_intervals.get(recording_session, None)
            
            # Process into snippets
            snippet_preds, snippet_labels = self.process_into_snippets(
                timestamps, predictions, labels,
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

    def evaluate_gcl_timestamps(self, predictions, labels):
        """
        Evaluate GCL timestamp-level predictions.
        """
        if predictions is None or labels is None:
            print("    Warning: No valid data for GCL evaluation")
            return None
        
        # Calculate basic metrics
        num_positive = int(labels.sum())
        num_negative = len(labels) - num_positive
        total_samples = len(labels)
        positive_rate = num_positive / total_samples if total_samples > 0 else 0
        
        print(f"    GCL snippet distribution: {num_positive} positive, {num_negative} negative (total: {total_samples} snippets, positive rate: {positive_rate:.3f})")
        
        # Calculate standard metrics
        metrics = calculate_binary_metrics(predictions, labels)
        
        # Calculate optimal threshold metrics
        optimal_metrics = self.calculate_optimal_threshold_metrics(predictions, labels)
        
        # Combine all metrics
        metrics.update(optimal_metrics)
        
        return metrics

    def calculate_optimal_threshold_metrics(self, predictions, labels):
        """
        Calculate metrics at optimal threshold for GCL evaluation.
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
                'optimal_accuracy': 1.0 if labels[0] == 0 else 1.0,
                'optimal_cohen_kappa': 0.0
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

            # Cohen's kappa (computed from confusion counts)
            n = tp + tn + fp + fn
            if n > 0:
                po = (tp + tn) / n
                pe = 0.0
                # Expected agreement for binary classification
                p_yes_true = (tp + fn) / n
                p_yes_pred = (tp + fp) / n
                p_no_true = (tn + fp) / n
                p_no_pred = (tn + fn) / n
                pe = p_yes_true * p_yes_pred + p_no_true * p_no_pred
                kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
            else:
                kappa = 0.0
            
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
                    'optimal_cohen_kappa': kappa,
                    'optimal_tp': int(tp),
                    'optimal_tn': int(tn),
                    'optimal_fp': int(fp),
                    'optimal_fn': int(fn)
                }
        
        return optimal_metrics

    def bootstrap_metrics(self, predictions, labels, n_bootstrap=100):
        """
        Calculate metrics using bootstrapping for GCL evaluation.
        """
        if len(predictions) == 0:
            return {}
        
        # Get all metric names
        metric_names = [
            'auc', 'average_precision', 'best_accuracy', 'best_precision', 'best_recall', 
            'best_f1', 'best_f2', 'best_f5', 'best_f10', 'best_cohen_kappa', 'positive_rate',
            'optimal_threshold', 'optimal_f1', 'optimal_f2', 'optimal_f5', 'optimal_f10',
            'optimal_precision', 'optimal_recall', 'optimal_accuracy', 'optimal_cohen_kappa'
        ]
        
        bootstrap_results = {metric: [] for metric in metric_names}
        
        # Bootstrap sampling
        for bootstrap_round in range(n_bootstrap):
            # Bootstrap samples with replacement
            n_samples = len(predictions)
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrapped_predictions = predictions[bootstrap_indices]
            bootstrapped_labels = labels[bootstrap_indices]
            
            # Calculate metrics for this bootstrapped sample
            bootstrapped_metrics = calculate_binary_metrics(bootstrapped_predictions, bootstrapped_labels)
            
            # Calculate optimal threshold metrics for bootstrapped sample
            optimal_metrics = self.calculate_optimal_threshold_metrics(bootstrapped_predictions, bootstrapped_labels)
            bootstrapped_metrics.update(optimal_metrics)
            
            # Store results
            for metric in metric_names:
                if metric in bootstrapped_metrics:
                    bootstrap_results[metric].append(bootstrapped_metrics[metric])
        
        # Calculate mean and std from bootstrap results
        result = {}
        for metric, values in bootstrap_results.items():
            if values:
                result[metric] = np.mean(values)
                result[f'{metric}_std'] = np.std(values)
        
        return result

    def run_full_evaluation(self, save_dir=None):
        """
        Run the complete GCL evaluation process.
        """

        
        # Discover files
        system_files = self.discover_gcl_system_files()
        human_files = self.discover_human_evaluation_files()
        

        
        # Aggregate human evaluations by timestamp
        aggregated_human_data = self.aggregate_human_evaluations_by_timestamp(human_files)
        

        # Match system and human data
        matched_data = self.match_system_and_human_data(system_files, aggregated_human_data)
        
        if not matched_data:
            print("Error: No matched data found!")
            return
        
        # Load and align all data
        predictions, labels, snippet_count = self.load_and_align_timestamp_data(matched_data, return_snippet_count=True)

        
        if predictions is None:
            return
        
        print(f"\nEvaluating GCL results:")
        print(f"  Total snippets: {snippet_count}")
        print(f"  Positive snippets: {int(labels.sum())}")
        print(f"  Negative snippets: {int((labels == 0).sum())}")
        print(f"  Positive rate: {labels.mean():.3f}")
        
        # Evaluate timestamps
        metrics = self.evaluate_gcl_timestamps(predictions, labels)
        
        if metrics is None:
            print("Error: Failed to calculate metrics!")
            return
        
        # Calculate bootstrap metrics for uncertainty estimation
        bootstrap_metrics = self.bootstrap_metrics(predictions, labels, n_bootstrap=100)
        
        # Combine metrics
        final_metrics = metrics.copy()
        final_metrics.update(bootstrap_metrics)
        
        # Store results
        self.timestamp_results = {
            'gcl_timestamps': final_metrics
        }
        
        self.summary_results = {
            'gcl_summary': {
                'micro_metrics': final_metrics,
                'num_snippets': snippet_count,
                'num_positives': int(labels.sum()),
                'num_negatives': int((labels == 0).sum()),
                'positive_rate': labels.mean()
            }
        }
        
        # Save results
        self.save_results(save_dir)
        
        # Generate visualization plots per matched session
        plots_dir = os.path.join(self.output_dir if save_dir is None else save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        for session, data in matched_data.items():
            try:
                # Reconstruct raw human series for plotting: build arrays from aggregated dict
                # Find the human session used for this matched session
                human_session = session
                if session.endswith('_aggregated'):
                    m = re.search(r'recording_(\d{4}_\d{2}_\d{2})_aggregated', session)
                    if m:
                        date_str = m.group(1)
                        # Find an aggregated human session that includes this date
                        # Reuse aggregation by merging matched_data labels at timestamps
                ts_sys = data['timestamps']
                ys_sys = data['predictions']
                ts_h = data['timestamps']
                ys_h = data['labels']
                plot_human_vs_system_timeseries(session, ts_sys, ys_sys, ts_h, ys_h, plots_dir)
            except Exception:
                continue
        
        final_save_dir = save_dir if save_dir else self.output_dir
        print(f"\nGCL Evaluation complete! Results saved to: {final_save_dir}")

    def save_results(self, save_dir=None):
        """Save GCL evaluation results."""
        if save_dir is None:
            save_dir = self.output_dir
        
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving GCL evaluation results to: {save_dir}")
        
        # Save timestamp results
        import json
        import pandas as pd
        
        # Save as JSON
        json_path = os.path.join(save_dir, 'gcl_timestamp_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.timestamp_results, f, indent=2)
        print(f"  Saved timestamp results to {json_path}")
        
        # Save as CSV
        csv_data = []
        for session, metrics in self.timestamp_results.items():
            row = {'session': session}
            row.update(metrics)
            csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(save_dir, 'gcl_timestamp_results.csv')
            df.to_csv(csv_path, index=False)
            print(f"  Saved timestamp results to {csv_path}")
        
        # Save summary results
        summary_json_path = os.path.join(save_dir, 'gcl_summary_results.json')
        with open(summary_json_path, 'w') as f:
            json.dump(self.summary_results, f, indent=2)
        print(f"  Saved summary results to {summary_json_path}")
        
        # Save summary as CSV
        summary_csv_data = []
        for session, summary in self.summary_results.items():
            if 'micro_metrics' in summary:
                row = {'session': session}
                row.update(summary['micro_metrics'])
                summary_csv_data.append(row)
        
        if summary_csv_data:
            df_summary = pd.DataFrame(summary_csv_data)
            summary_csv_path = os.path.join(save_dir, 'gcl_summary_results.csv')
            df_summary.to_csv(summary_csv_path, index=False)
            print(f"  Saved summary results to {summary_csv_path}")
