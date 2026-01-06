import os
import numpy as np
from datetime import datetime

def generate_detailed_report(identity_results, summary_results, save_dir):
    print("Generating detailed text report...")
    report_path = os.path.join(save_dir, 'detailed_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE HUMAN vs SYSTEM EVALUATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Evaluation method: 20-second snippet-based evaluation\n")
        f.write(f"Data source: Filtered human evaluation results\n")
        f.write(f"Positive label criteria: (results > 0) AND (count > 0)\n")
        f.write("\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 40 + "\n")
        for split in ['train', 'val', 'test']:
            if split in summary_results:
                metrics = summary_results[split]['macro_metrics']
                f.write(f"\n{split.upper()} SPLIT:\n")
                f.write(f"  Total samples: {metrics['num_samples']:,}\n")
                f.write(f"  Positive samples: {metrics['num_positives']:,}\n")
                f.write(f"  Positive rate: {metrics['positive_rate']:.4f}\n")
                f.write(f"  Number of identities: {summary_results[split]['num_identities']}\n")
                f.write(f"  F1 Score: {metrics['best_f1']:.4f}\n")
                f.write(f"  F2 Score: {metrics['best_f2']:.4f}\n")
                f.write(f"  Precision: {metrics['best_precision']:.4f}\n")
                f.write(f"  Recall: {metrics['best_recall']:.4f}\n")
                f.write(f"  AUC: {metrics['auc']:.4f}\n")
                f.write(f"  Average Precision: {metrics['average_precision']:.4f}\n")
            else:
                f.write(f"\n{split.upper()} SPLIT: No data available\n")
        # Add Top-N metrics summary for each split
        for split in ['train', 'val', 'test']:
            if split in summary_results and 'topN_metrics' in summary_results[split]:
                f.write(f"\n{split.upper()} TOP-N METRICS (mean ± std):\n")
                topN = summary_results[split]['topN_metrics']
                for N in sorted(topN.keys()):
                    metrics = topN[N]
                    f.write(f"  Top-{N}:\n")
                    for metric in [
                        'auc', 'average_precision', 'best_accuracy', 'best_precision', 'best_recall',
                        'best_f1', 'best_f2', 'best_f5', 'best_f10']:
                        mean = metrics.get(f'{metric}@{N}', None)
                        std = metrics.get(f'{metric}@{N}_std', None)
                        if mean is not None and std is not None:
                            f.write(f"    {metric}: {mean:.4f} ± {std:.4f}\n")
        f.write("\n\nDETAILED RESULTS BY IDENTITY\n")
        f.write("-" * 40 + "\n")
        for split in ['train', 'val', 'test']:
            if split in identity_results and identity_results[split]:
                f.write(f"\n{split.upper()} SPLIT IDENTITIES:\n")
                identity_data = identity_results[split]
                sorted_identities = sorted(identity_data.items(), 
                                         key=lambda x: x[1]['best_f2'], reverse=True)
                for identity, metrics in sorted_identities:
                    f.write(f"\n  {identity}:\n")
                    f.write(f"    Samples: {metrics['num_samples']:,} (pos: {metrics['num_positives']:,})\n")
                    f.write(f"    F1: {metrics['best_f1']:.4f} | F2: {metrics['best_f2']:.4f}\n")
                    f.write(f"    F5: {metrics.get('best_f5', 0):.4f} | F10: {metrics.get('best_f10', 0):.4f}\n")
                    f.write(f"    Precision: {metrics['best_precision']:.4f} | Recall: {metrics['best_recall']:.4f}\n")
                    f.write(f"    AUC: {metrics['auc']:.4f} | AP: {metrics['average_precision']:.4f}\n")
                    f.write(f"    Threshold (F1): {metrics.get('best_threshold_f1', 0.5):.3f}\n")
                    f.write(f"    Threshold (F2): {metrics.get('best_threshold_f2', 0.5):.3f}\n")
                    f.write(f"    Threshold (F5): {metrics.get('best_threshold_f5', 0.5):.3f}\n")
                    f.write(f"    Threshold (F10): {metrics.get('best_threshold_f10', 0.5):.3f}\n")
                    f.write(f"    Confusion Matrix: TP={metrics['true_positives']}, TN={metrics['true_negatives']}, FP={metrics['false_positives']}, FN={metrics['false_negatives']}\n")
        f.write("\n\nPERFORMACE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("\nF2 SCORE ANALYSIS:\n")
        f.write("F2 score emphasizes recall over precision with a 2:1 ratio.\n")
        f.write("Formula: F2 = 5 * precision * recall / (4 * precision + recall)\n\n")
        for split in ['train', 'val', 'test']:
            if split in identity_results and identity_results[split]:
                identity_data = identity_results[split]
                f2_scores = [metrics['best_f2'] for metrics in identity_data.values()]
                f.write(f"{split.upper()} F2 Statistics:\n")
                f.write(f"  Mean: {np.mean(f2_scores):.4f}\n")
                f.write(f"  Std: {np.std(f2_scores):.4f}\n")
                f.write(f"  Min: {np.min(f2_scores):.4f}\n")
                f.write(f"  Max: {np.max(f2_scores):.4f}\n")
                f.write(f"  Median: {np.median(f2_scores):.4f}\n\n")
        f.write("TOP PERFORMERS BY F2 SCORE:\n")
        all_performances = []
        for split in ['train', 'val', 'test']:
            if split in identity_results and identity_results[split]:
                identity_data = identity_results[split]
                for identity, metrics in identity_data.items():
                    all_performances.append((f"{identity}_{split}", metrics['best_f2'], 
                                           metrics['best_precision'], metrics['best_recall']))
        if all_performances:
            top_performers = sorted(all_performances, key=lambda x: x[1], reverse=True)[:10]
            for i, (name, f2, prec, rec) in enumerate(top_performers, 1):
                f.write(f"  {i:2d}. {name}: F2={f2:.4f} (P={prec:.4f}, R={rec:.4f})\n")
            f.write("\nBOTTOM PERFORMERS BY F2 SCORE:\n")
            bottom_performers = sorted(all_performances, key=lambda x: x[1])[:10]
            for i, (name, f2, prec, rec) in enumerate(bottom_performers, 1):
                f.write(f"  {i:2d}. {name}: F2={f2:.4f} (P={prec:.4f}, R={rec:.4f})\n")
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    print(f"Detailed report saved to {report_path}") 