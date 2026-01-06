import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, cohen_kappa_score


def calculate_f2_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 5 * precision * recall / (4 * precision + recall)

def calculate_fbeta_score(precision, recall, beta):
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

def calculate_f5_score(precision, recall):
    return calculate_fbeta_score(precision, recall, 5)

def calculate_f10_score(precision, recall):
    return calculate_fbeta_score(precision, recall, 10)

def non_boot_calculate_binary_metrics(predictions, labels):
    """
    Calculate binary classification metrics without bootstrapping.
    """
    metrics = {}
    labels = (labels > 0).astype(int)
    
    if len(np.unique(labels)) == 1:
        if labels[0] == 0:
            metrics['auc'] = 0.5
            metrics['average_precision'] = 0.0
            metrics['cohen_kappa'] = 0.0
            metrics.update({
                'best_accuracy': 1.0,
                'best_precision': 0.0,
                'best_recall': 0.0,
                'best_f1': 0.0,
                'best_f2': 0.0,
                'best_f5': 0.0,
                'best_f10': 0.0,
                'true_positives': 0,
                'true_negatives': len(labels),
                'false_positives': 0,
                'false_negatives': 0
            })
            return metrics
        else:
            metrics['auc'] = 1.0
            metrics['average_precision'] = 1.0
            metrics['cohen_kappa'] = 1.0
            metrics.update({
                'best_accuracy': 1.0,
                'best_precision': 1.0,
                'best_recall': 1.0,
                'best_f1': 1.0,
                'best_f2': 1.0,
                'best_f5': 1.0,
                'best_f10': 1.0,
                'true_positives': len(labels),
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0
            })
            return metrics
    else:
        try:
            metrics['auc'] = roc_auc_score(labels, predictions)
            metrics['average_precision'] = average_precision_score(labels, predictions)
        except Exception as e:
            metrics['auc'] = 0.5
            metrics['average_precision'] = 0.0
    
    # Find optimal threshold using F1-score maximization
    thresholds = np.linspace(0, 1, 21)
    best_f1 = 0
    optimal_threshold = 0.5  # default
    
    for threshold in thresholds:
        binary_preds = (predictions >= threshold).astype(int)
        try:
            f1 = f1_score(labels, binary_preds, average='micro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                optimal_threshold = threshold
        except:
            continue
    
    # If F1 is still 0, try to find a threshold that gives at least some positive predictions
    if best_f1 == 0:
        best_score = 0
        for threshold in thresholds:
            binary_preds = (predictions >= threshold).astype(int)
            try:
                precision = precision_score(labels, binary_preds, average='micro', zero_division=0)
                recall = recall_score(labels, binary_preds, average='micro', zero_division=0)
                
                # Prefer thresholds that give some positive predictions
                if np.sum(binary_preds) > 0:
                    score = recall  # Prioritize recall when we have few positive samples
                    if score > best_score:
                        best_score = score
                        optimal_threshold = threshold
            except:
                continue
    
    # Calculate all metrics at the optimal threshold
    binary_preds = (predictions >= optimal_threshold).astype(int)
    
    # Calculate confusion matrix manually to ensure correctness
    tp = np.sum((labels == 1) & (binary_preds == 1))
    tn = np.sum((labels == 0) & (binary_preds == 0))
    fp = np.sum((labels == 0) & (binary_preds == 1))
    fn = np.sum((labels == 1) & (binary_preds == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F-beta scores using the proper formula
    f1_score_val = calculate_fbeta_score(precision, recall, 1)
    f2_score_val = calculate_f2_score(precision, recall)
    f5_score_val = calculate_f5_score(precision, recall)
    f10_score_val = calculate_f10_score(precision, recall)
    
    # Cohen's kappa
    cohen_kappa_val = cohen_kappa_score(labels, binary_preds)
    
    metrics.update({
        'best_f1': f1_score_val,
        'best_f2': f2_score_val,
        'best_f5': f5_score_val,
        'best_f10': f10_score_val,
        'best_accuracy': accuracy,
        'best_precision': precision,
        'best_recall': recall,
        'best_cohen_kappa': cohen_kappa_val,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    })
    
    return metrics

def calculate_binary_metrics(predictions, labels):
    """
    Calculate binary classification metrics without bootstrapping.
    
    Args:
        predictions: System predictions
        labels: Ground truth labels
        
    Returns:
        dict: Metrics calculated from the given predictions and labels
    """
    # Use the non-bootstrap version directly
    metrics = non_boot_calculate_binary_metrics(predictions, labels)
    
    # Add sample information
    metrics['num_samples'] = len(predictions)
    metrics['num_positives'] = int(labels.sum())
    metrics['positive_rate'] = labels.mean()
    
    return metrics

 