import numpy as np
from .metrics import calculate_binary_metrics

def calculate_optimal_threshold_metrics(predictions, labels):
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

def bootstrap_metrics(predictions, labels, n_bootstrap=100):
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
        'optimal_precision', 'optimal_recall', 'optimal_accuracy'
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
        optimal_metrics = calculate_optimal_threshold_metrics(bootstrapped_predictions, bootstrapped_labels)
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

def evaluate_gcl_timestamps(predictions, labels):
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
    optimal_metrics = calculate_optimal_threshold_metrics(predictions, labels)
    
    # Combine all metrics
    metrics.update(optimal_metrics)
    
    return metrics
