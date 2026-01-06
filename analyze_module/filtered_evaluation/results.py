import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .plots import plot_confusion_matrices, plot_recall_by_identity, plot_f2_score_analysis
from .report import generate_detailed_report


def save_individual_identity_results(identity_results, save_dir):
    """
    Save individual identity results and visualizations in separate folders.
    
    Args:
        identity_results: Dictionary of identity results by split
        save_dir: Base directory to save results
    """
    identity_dir = os.path.join(save_dir, 'identity')
    os.makedirs(identity_dir, exist_ok=True)
    print(f"Saving individual identity results to: {identity_dir}")
    
    # Collect all unique identities across all splits
    all_identities = set()
    for split in ['train', 'val', 'test']:
        if split in identity_results and identity_results[split]:
            all_identities.update(identity_results[split].keys())
    
    for identity in sorted(all_identities):
        identity_folder = os.path.join(identity_dir, identity)
        os.makedirs(identity_folder, exist_ok=True)
        
        # Collect results for this identity across all splits
        identity_data = {}
        for split in ['train', 'val', 'test']:
            if split in identity_results and identity in identity_results[split]:
                identity_data[split] = identity_results[split][identity]
        
        if not identity_data:
            continue
            
        # Save quantitative results
        save_identity_quantitative_results(identity, identity_data, identity_folder)
        
        # Create visualizations
        create_identity_visualizations(identity, identity_data, identity_folder)
        
        print(f"  ✓ Saved results for {identity}")


def save_identity_quantitative_results(identity, identity_data, identity_folder):
    """
    Save quantitative results for a specific identity.
    
    Args:
        identity: Identity name
        identity_data: Dictionary of results by split
        identity_folder: Folder to save results
    """
    # Save detailed metrics CSV
    results_list = []
    for split, metrics in identity_data.items():
        row = {'identity': identity, 'split': split}
        row.update(metrics)
        results_list.append(row)
    
    df = pd.DataFrame(results_list)
    csv_path = os.path.join(identity_folder, f'{identity}_metrics.csv')
    df.to_csv(csv_path, index=False)
    
    # Save summary JSON
    summary_data = {
        'identity': identity,
        'splits': list(identity_data.keys()),
        'metrics': identity_data
    }
    json_path = os.path.join(identity_folder, f'{identity}_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Create summary text file
    txt_path = os.path.join(identity_folder, f'{identity}_summary.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Identity: {identity}\n")
        f.write("=" * 50 + "\n\n")
        
        for split, metrics in identity_data.items():
            f.write(f"{split.upper()} SPLIT:\n")
            f.write(f"  Samples: {metrics['num_samples']:,} (positives: {metrics['num_positives']:,})\n")
            f.write(f"  F1 Score: {metrics['best_f1']:.4f}\n")
            f.write(f"  F2 Score: {metrics['best_f2']:.4f}\n")
            f.write(f"  F5 Score: {metrics.get('best_f5', 0):.4f}\n")
            f.write(f"  F10 Score: {metrics.get('best_f10', 0):.4f}\n")
            f.write(f"  Precision: {metrics['best_precision']:.4f}\n")
            f.write(f"  Recall: {metrics['best_recall']:.4f}\n")
            f.write(f"  AUC: {metrics['auc']:.4f}\n")
            f.write(f"  Average Precision: {metrics['average_precision']:.4f}\n")
            f.write(f"  Accuracy: {metrics['best_accuracy']:.4f}\n")
            if 'best_cohen_kappa' in metrics:
                f.write(f"  Cohen Kappa: {metrics['best_cohen_kappa']:.4f}\n")
            f.write(f"  Confusion Matrix: TP={metrics['true_positives']}, TN={metrics['true_negatives']}, FP={metrics['false_positives']}, FN={metrics['false_negatives']}\n")
            
            # Add optimal threshold information
            if 'optimal_threshold' in metrics:
                f.write(f"\n  OPTIMAL THRESHOLD ANALYSIS:\n")
                f.write(f"    Optimal Threshold: {metrics['optimal_threshold']:.4f}\n")
                f.write(f"    Optimal F1: {metrics['optimal_f1']:.4f}\n")
                f.write(f"    Optimal F2: {metrics['optimal_f2']:.4f}\n")
                f.write(f"    Optimal F5: {metrics['optimal_f5']:.4f}\n")
                f.write(f"    Optimal F10: {metrics['optimal_f10']:.4f}\n")
                f.write(f"    Optimal Precision: {metrics['optimal_precision']:.4f}\n")
                f.write(f"    Optimal Recall: {metrics['optimal_recall']:.4f}\n")
                f.write(f"    Optimal Accuracy: {metrics['optimal_accuracy']:.4f}\n")
                f.write(f"    Optimal Confusion Matrix: TP={metrics['optimal_tp']}, TN={metrics['optimal_tn']}, FP={metrics['optimal_fp']}, FN={metrics['optimal_fn']}\n")
            f.write("\n")


def create_identity_visualizations(identity, identity_data, identity_folder):
    """
    Create visualizations for a specific identity.
    
    Args:
        identity: Identity name
        identity_data: Dictionary of results by split
        identity_folder: Folder to save visualizations
    """
    # 1. Metrics comparison across splits
    create_identity_metrics_comparison(identity, identity_data, identity_folder)
    
    # 2. Confusion matrix for each split
    create_identity_confusion_matrices(identity, identity_data, identity_folder)
    
    # 3. Performance radar chart
    create_identity_radar_chart(identity, identity_data, identity_folder)
    
    # 4. Optimal threshold analysis
    create_identity_optimal_threshold_analysis(identity, identity_data, identity_folder)


def create_metric_std_scatter_plots(identity_results, save_dir):
    """
    Create scatter plots showing metric values vs std for all identities.
    
    Args:
        identity_results: Dictionary of identity results by split
        save_dir: Directory to save plots
    """
    # Collect all metrics and their std values across all identities and splits
    all_metrics_data = {}
    
    for split in ['train', 'val', 'test']:
        if split in identity_results and identity_results[split]:
            for identity, metrics in identity_results[split].items():
                for metric_name, value in metrics.items():
                    if metric_name.endswith('_std'):
                        base_metric = metric_name[:-4]  # Remove '_std' suffix
                        if base_metric in metrics:
                            if base_metric not in all_metrics_data:
                                all_metrics_data[base_metric] = {'values': [], 'stds': [], 'identities': [], 'splits': []}
                            all_metrics_data[base_metric]['values'].append(metrics[base_metric])
                            all_metrics_data[base_metric]['stds'].append(value)
                            all_metrics_data[base_metric]['identities'].append(identity)
                            all_metrics_data[base_metric]['splits'].append(split)
    
    # Create scatter plots for each metric
    metrics_to_plot = [
        'auc', 'average_precision', 'best_accuracy', 'best_precision', 'best_recall',
        'best_f1', 'best_f2', 'best_f5', 'best_f10', 'best_cohen_kappa',
        'optimal_threshold', 'optimal_f1', 'optimal_f2', 'optimal_f5', 'optimal_f10',
        'optimal_precision', 'optimal_recall', 'optimal_accuracy'
    ]
    
    # Filter metrics that have data
    available_metrics = [metric for metric in metrics_to_plot if metric in all_metrics_data and len(all_metrics_data[metric]['values']) > 0]
    
    if not available_metrics:
        print("  No metric data available for scatter plots")
        return
    
    # Calculate number of rows and columns for subplot grid
    n_metrics = len(available_metrics)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    colors = {'train': 'blue', 'val': 'green', 'test': 'red'}
    
    for i, metric in enumerate(available_metrics):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        data = all_metrics_data[metric]
        values = np.array(data['values'])
        stds = np.array(data['stds'])
        splits = data['splits']
        
        # Create scatter plot with different colors for each split
        for split in ['train', 'val', 'test']:
            mask = [s == split for s in splits]
            if any(mask):
                ax.scatter(np.array(values)[mask], np.array(stds)[mask], 
                          c=colors[split], label=split, alpha=0.6, s=30)
        
        ax.set_xlabel(f'{metric.replace("_", " ").title()}')
        ax.set_ylabel('Standard Deviation')
        ax.set_title(f'{metric.replace("_", " ").title()} vs Std')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(values) > 1:
            correlation = np.corrcoef(values, stds)[0, 1]
            ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Hide unused subplots
    for i in range(len(available_metrics), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('Metric Values vs Standard Deviation Across All Identities', fontsize=16)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'metric_std_scatter_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved metric vs std scatter plots to {plot_path}")


def create_topN_metric_violin_plots(identity_results, save_dir, N_values=[3, 5, 7], n_bootstrap=10):
    """
    Create violin plots for each metric using bootstrap samples over Top-N identities.
    For each split and each N in N_values, we:
      - select top-N identities by best_f2
      - run bootstrap over identities (sample N with replacement) for n_bootstrap rounds
      - compute the mean of each metric per bootstrap round
      - plot a violin (distribution) per metric

    Args:
        identity_results: Dict[split][identity] -> metrics dict
        save_dir: Directory to save plots
        N_values: List of N values for Top-N
        n_bootstrap: Number of bootstrap rounds
    """
    rng = np.random.default_rng(42)
    metrics_to_plot = [
        'auc', 'average_precision', 'best_accuracy', 'best_precision', 'best_recall',
        'best_f1', 'best_f2', 'best_f5', 'best_f10', 'best_cohen_kappa',
        'optimal_threshold', 'optimal_f1', 'optimal_f2', 'optimal_f5', 'optimal_f10',
        'optimal_precision', 'optimal_recall', 'optimal_accuracy'
    ]

    for split in ['train', 'val', 'test']:
        if split not in identity_results or not identity_results[split]:
            continue

        # Sort identities by best_f2 desc
        identities_items = list(identity_results[split].items())
        identities_items = [item for item in identities_items if 'best_f2' in item[1]]
        identities_items.sort(key=lambda x: x[1]['best_f2'], reverse=True)

        for N in N_values:
            if not identities_items:
                continue
            topK = identities_items[:min(N, len(identities_items))]
            top_ids = [iid for iid, _ in topK]
            top_metrics = {iid: identity_results[split][iid] for iid in top_ids}

            # Prepare bootstrap distributions per metric
            distributions = {m: [] for m in metrics_to_plot}

            # Determine which metrics are available
            available_metrics = []
            for m in metrics_to_plot:
                if any(m in md for md in top_metrics.values()):
                    available_metrics.append(m)

            if not available_metrics:
                print(f"  No metrics available for violin plots (Top-{N}, {split})")
                continue

            # Bootstrap over identities
            id_list = list(top_metrics.keys())
            size_k = len(id_list)
            for _ in range(n_bootstrap):
                sampled_ids = rng.choice(id_list, size=size_k, replace=True)
                # Compute mean metric over sampled identities
                for m in available_metrics:
                    vals = [top_metrics[iid][m] for iid in sampled_ids if m in top_metrics[iid]]
                    if vals:
                        distributions[m].append(float(np.mean(vals)))

            # Filter distributions to only those with samples
            plot_metrics = [m for m in available_metrics if len(distributions[m]) > 0]
            if not plot_metrics:
                print(f"  No bootstrap samples for violin plots (Top-{N}, {split})")
                continue

            # Build violin plot figure
            fig, ax = plt.subplots(figsize=(16, 10))
            violin_data = [distributions[m] for m in plot_metrics]
            violin_labels = [m.replace('_', ' ').title() for m in plot_metrics]

            violin_parts = ax.violinplot(violin_data, showmeans=True, showmedians=True)
            if 'cmeans' in violin_parts:
                violin_parts['cmeans'].set_color('red')
                violin_parts['cmeans'].set_linewidth(2)
            if 'cmedians' in violin_parts:
                violin_parts['cmedians'].set_color('blue')
                violin_parts['cmedians'].set_linewidth(2)
            for pc in violin_parts['bodies']:
                pc.set_facecolor('lightblue')
                pc.set_alpha(0.7)

            # Annotate mean/median per violin
            for i, m in enumerate(plot_metrics):
                vals = violin_data[i]
                mean_val = float(np.mean(vals)) if vals else 0.0
                median_val = float(np.median(vals)) if vals else 0.0
                ax.text(i + 1, mean_val, f'μ={mean_val:.3f}',
                        ha='center', va='bottom', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                ax.text(i + 1, median_val, f'med={median_val:.3f}',
                        ha='center', va='top', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            ax.set_xlabel('Metrics')
            ax.set_ylabel('Bootstrap Mean Values')
            ax.set_title(f'Distribution of Top-{N} Metrics via Bootstrap - {split.upper()} Split')
            ax.set_xticks(range(1, len(plot_metrics) + 1))
            ax.set_xticklabels(violin_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', lw=2, label='Mean'),
                Line2D([0], [0], color='blue', lw=2, label='Median')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            plt.tight_layout()
            plot_path = os.path.join(save_dir, f'metric_violin_plots_top{N}_{split}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved Top-{N} {split} violin plots to {plot_path}")

def create_metric_violin_plots(identity_results, save_dir):
    """
    Create violin plots for each metric by split, with separate figures for each split.
    
    Args:
        identity_results: Dictionary of identity results by split
        save_dir: Directory to save plots
    """
    # Define metrics to plot
    metrics_to_plot = [
        'auc', 'average_precision', 'best_accuracy', 'best_precision', 'best_recall',
        'best_f1', 'best_f2', 'best_f5', 'best_f10', 'best_cohen_kappa',
        'optimal_threshold', 'optimal_f1', 'optimal_f2', 'optimal_f5', 'optimal_f10',
        'optimal_precision', 'optimal_recall', 'optimal_accuracy'
    ]
    
    # Create separate violin plot for each split
    for split in ['train', 'val', 'test']:
        if split not in identity_results or not identity_results[split]:
            continue
            
        # Collect data for this split
        split_data = {}
        available_metrics = []
        
        for metric in metrics_to_plot:
            values = []
            for identity, metrics_dict in identity_results[split].items():
                if metric in metrics_dict:
                    values.append(metrics_dict[metric])
            
            if values:  # Only include metrics that have data
                split_data[metric] = values
                available_metrics.append(metric)
        
        if not available_metrics:
            print(f"  No metric data available for {split} violin plots")
            continue
        
        # Create violin plot
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Prepare data for violin plot
        violin_data = []
        violin_labels = []
        
        for metric in available_metrics:
            violin_data.append(split_data[metric])
            violin_labels.append(metric.replace('_', ' ').title())
        
        # Create violin plot
        violin_parts = ax.violinplot(violin_data, showmeans=True, showmedians=True)
        
        # Customize violin plot appearance
        violin_parts['cmeans'].set_color('red')
        violin_parts['cmeans'].set_linewidth(2)
        violin_parts['cmedians'].set_color('blue')
        violin_parts['cmedians'].set_linewidth(2)
        
        # Color the violin bodies
        for pc in violin_parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        # Add statistics
        for i, metric in enumerate(available_metrics):
            values = split_data[metric]
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)
            
            # Add text annotations
            ax.text(i + 1, mean_val, f'μ={mean_val:.3f}', 
                   ha='center', va='bottom', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            ax.text(i + 1, median_val, f'med={median_val:.3f}', 
                   ha='center', va='top', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Customize plot
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Metric Values')
        ax.set_title(f'Distribution of Metrics Across All Identities - {split.upper()} Split')
        ax.set_xticks(range(1, len(available_metrics) + 1))
        ax.set_xticklabels(violin_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Mean'),
            Line2D([0], [0], color='blue', lw=2, label='Median')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, f'metric_violin_plots_{split}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved {split} violin plots to {plot_path}")


def create_sample_distribution_plots(identity_results, save_dir):
    """
    Create stacked bar plots showing positive and negative sample counts for each identity.
    
    Args:
        identity_results: Dictionary of identity results by split
        save_dir: Directory to save plots
    """
    # Create separate plot for each split
    for split in ['train', 'val', 'test']:
        if split not in identity_results or not identity_results[split]:
            continue
        
        # Collect data for this split
        identities = []
        positive_counts = []
        negative_counts = []
        total_counts = []
        
        for identity, metrics in identity_results[split].items():
            if 'num_positives' in metrics and 'num_samples' in metrics:
                identities.append(identity)
                positive_count = metrics['num_positives']
                total_count = metrics['num_samples']
                negative_count = total_count - positive_count
                
                positive_counts.append(positive_count)
                negative_counts.append(negative_count)
                total_counts.append(total_count)
        
        if not identities:
            print(f"  No sample data available for {split}")
            continue
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(max(12, len(identities) * 0.8), 8))
        
        # Set up bar positions
        x_pos = np.arange(len(identities))
        bar_width = 0.6
        
        # Create stacked bar plot
        # Positive samples (top)
        positive_bars = ax.bar(x_pos, positive_counts, bar_width, 
                              label='Positive Samples', color='#ff7f0e', alpha=0.8)
        
        # Negative samples (bottom) - stacked on top of positive
        negative_bars = ax.bar(x_pos, negative_counts, bar_width, 
                              bottom=positive_counts, label='Negative Samples', 
                              color='#1f77b4', alpha=0.8)
        
        # Add value labels on bars
        for i, (pos_count, neg_count, total_count) in enumerate(zip(positive_counts, negative_counts, total_counts)):
            # Label for positive samples (top)
            if pos_count > 0:
                ax.text(x_pos[i], pos_count/2, f'{pos_count}', 
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       color='white')
            
            # Label for negative samples (bottom)
            if neg_count > 0:
                ax.text(x_pos[i], pos_count + neg_count/2, f'{neg_count}', 
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       color='white')
            
            # Total count label at the top
            ax.text(x_pos[i], total_count + max(total_counts) * 0.02, f'{total_count}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Customize plot
        ax.set_xlabel('Identities', fontsize=14)
        ax.set_ylabel('Number of Samples', fontsize=14)
        ax.set_title(f'Sample Distribution by Identity - {split.upper()} Split', fontsize=16, pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(identities, rotation=45, ha='right', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        ax.legend(loc='upper right', fontsize=12)
        
        # Set y-axis limits with some padding
        max_total = max(total_counts) if total_counts else 100
        ax.set_ylim(0, max_total * 1.1)
        
        # Add percentage annotations
        for i, (pos_count, neg_count, total_count) in enumerate(zip(positive_counts, negative_counts, total_counts)):
            if total_count > 0:
                pos_percentage = (pos_count / total_count) * 100
                neg_percentage = (neg_count / total_count) * 100
                
                # Add percentage text below x-axis labels
                ax.text(x_pos[i], -max_total * 0.05, f'{pos_percentage:.1f}%', 
                       ha='center', va='top', fontsize=9, color='#ff7f0e', fontweight='bold')
                ax.text(x_pos[i], -max_total * 0.1, f'{neg_percentage:.1f}%', 
                       ha='center', va='top', fontsize=9, color='#1f77b4', fontweight='bold')
        
        # Adjust layout to accommodate percentage labels
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, f'sample_distribution_{split}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved sample distribution plot for {split} to {plot_path}")


def create_identity_metrics_comparison(identity, identity_data, identity_folder):
    """Create bar chart comparing metrics across splits."""
    splits = list(identity_data.keys())
    metrics_to_plot = ['best_f1', 'best_f2', 'best_f5', 'best_f10', 'best_precision', 'best_recall', 'auc']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        if i >= len(axes):
            break
            
        values = [identity_data[split].get(metric, 0) for split in splits]
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        
        bars = axes[i].bar(splits, values, color=colors[:len(splits)])
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Score')
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    # Hide unused subplots
    for i in range(len(metrics_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{identity} - Metrics Comparison Across Splits', fontsize=16)
    plt.tight_layout()
    
    plot_path = os.path.join(identity_folder, f'{identity}_metrics_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_identity_confusion_matrices(identity, identity_data, identity_folder):
    """Create separate confusion matrix images for each split."""
    for split in identity_data.keys():
        metrics = identity_data[split]
        cm = np.array([
            [metrics['true_negatives'], metrics['false_positives']],
            [metrics['false_negatives'], metrics['true_positives']]
        ])
        
        # Create single plot for this split
        fig, ax = plt.subplots(figsize=(6, 5))
        
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', ax=ax,
                   xticklabels=['Predicted Negative', 'Predicted Positive'],
                   yticklabels=['Actual Negative', 'Actual Positive'])
        
        f1_score = metrics['best_f1']
        f2_score = metrics['best_f2']
        precision = metrics['best_precision']
        recall = metrics['best_recall']
        
        ax.set_title(f'{identity} - Confusion Matrices\n'
                    f'F1: {f1_score:.3f} | F2: {f2_score:.3f}\n'
                    f'Precision: {precision:.3f} | Recall: {recall:.3f}')
        
        plt.tight_layout()
        
        # Save with split in filename but same title
        plot_path = os.path.join(identity_folder, f'{identity}_confusion_matrix_{split}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_identity_radar_chart(identity, identity_data, identity_folder):
    """Create radar chart showing performance metrics."""
    # Use the first available split for radar chart
    split = list(identity_data.keys())[0]
    metrics = identity_data[split]
    
    # Define metrics for radar chart
    radar_metrics = ['best_f1', 'best_f2', 'best_precision', 'best_recall', 'auc', 'best_accuracy']
    metric_labels = ['F1', 'F2', 'Precision', 'Recall', 'AUC', 'Accuracy']
    
    values = [metrics.get(metric, 0) for metric in radar_metrics]
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    values += values[:1]  # Close the plot
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, label=identity)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title(f'{identity} - Performance Radar Chart ({split.upper()})', pad=20)
    ax.grid(True)
    
    plot_path = os.path.join(identity_folder, f'{identity}_radar_chart.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_identity_optimal_threshold_analysis(identity, identity_data, identity_folder):
    """Create visualization for optimal threshold analysis."""
    splits = list(identity_data.keys())
    
    # Create subplot for optimal threshold comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # 1. Optimal threshold values
    optimal_thresholds = [identity_data[split].get('optimal_threshold', 0.5) for split in splits]
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    bars1 = axes[0].bar(splits, optimal_thresholds, color=colors[:len(splits)])
    axes[0].set_title('Optimal Threshold Values')
    axes[0].set_ylabel('Threshold')
    axes[0].set_ylim(0, 1)
    for bar, value in zip(bars1, optimal_thresholds):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # 2. Optimal F-scores comparison
    optimal_f1 = [identity_data[split].get('optimal_f1', 0) for split in splits]
    optimal_f2 = [identity_data[split].get('optimal_f2', 0) for split in splits]
    optimal_f5 = [identity_data[split].get('optimal_f5', 0) for split in splits]
    optimal_f10 = [identity_data[split].get('optimal_f10', 0) for split in splits]
    
    x = np.arange(len(splits))
    width = 0.2
    
    axes[1].bar(x - 1.5*width, optimal_f1, width, label='F1', alpha=0.8)
    axes[1].bar(x - 0.5*width, optimal_f2, width, label='F2', alpha=0.8)
    axes[1].bar(x + 0.5*width, optimal_f5, width, label='F5', alpha=0.8)
    axes[1].bar(x + 1.5*width, optimal_f10, width, label='F10', alpha=0.8)
    
    axes[1].set_title('Optimal F-Scores Comparison')
    axes[1].set_ylabel('Score')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits)
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    
    # 3. Optimal Precision vs Recall
    optimal_precision = [identity_data[split].get('optimal_precision', 0) for split in splits]
    optimal_recall = [identity_data[split].get('optimal_recall', 0) for split in splits]
    
    axes[2].scatter(optimal_recall, optimal_precision, c=colors[:len(splits)], s=100, alpha=0.7)
    for i, split in enumerate(splits):
        axes[2].annotate(split, (optimal_recall[i], optimal_precision[i]), 
                        xytext=(5, 5), textcoords='offset points')
    
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Optimal Precision vs Recall')
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    
    # 4. Optimal vs Best metrics comparison
    best_f2 = [identity_data[split].get('best_f2', 0) for split in splits]
    optimal_f2 = [identity_data[split].get('optimal_f2', 0) for split in splits]
    
    x = np.arange(len(splits))
    width = 0.35
    
    bars3 = axes[3].bar(x - width/2, best_f2, width, label='Best F2', alpha=0.8)
    bars4 = axes[3].bar(x + width/2, optimal_f2, width, label='Optimal F2', alpha=0.8)
    
    axes[3].set_title('Best vs Optimal F2 Score')
    axes[3].set_ylabel('F2 Score')
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(splits)
    axes[3].legend()
    axes[3].set_ylim(0, 1)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        axes[3].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        axes[3].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'{identity} - Optimal Threshold Analysis', fontsize=16)
    plt.tight_layout()
    
    plot_path = os.path.join(identity_folder, f'{identity}_optimal_threshold_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def write_evaluation_results(identity_results, summary_results, save_dir, skipped_evaluations=None):
    if save_dir is None:
        raise ValueError("save_dir must be provided")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving comprehensive evaluation results to: {save_dir}")

    # Save individual identity results
    save_individual_identity_results(identity_results, save_dir)

    # Create metric vs std scatter plots for all identities
    create_metric_std_scatter_plots(identity_results, save_dir)

    # Create metric violin plots for each split
    create_metric_violin_plots(identity_results, save_dir)

    # Create sample distribution plots
    create_sample_distribution_plots(identity_results, save_dir)

    # Save identity-level results
    for split in ['train', 'val', 'test']:
        if split in identity_results and identity_results[split]:
            results_list = []
            for identity, metrics in identity_results[split].items():
                result_row = {'identity': identity, 'split': split}
                result_row.update(metrics)
                results_list.append(result_row)
            df = pd.DataFrame(results_list)
            output_file = os.path.join(save_dir, f'identity_results_{split}.csv')
            df.to_csv(output_file, index=False)
            print(f"  Saved {split} identity results to {output_file}")

    # Save summary json/csv
    summary_output = os.path.join(save_dir, 'summary_results.json')
    with open(summary_output, 'w') as f:
        json.dump(summary_results, f, indent=2)
    print(f"  Saved summary results to {summary_output}")

    summary_csv_output = os.path.join(save_dir, 'summary_results.csv')
    summary_rows = []
    for split, split_results in summary_results.items():
        for metric_type, metrics in split_results.items():
            if isinstance(metrics, dict):
                row = {'split': split, 'metric_type': metric_type}
                row.update(metrics)
                summary_rows.append(row)
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(summary_csv_output, index=False)
        print(f"  Saved summary results to {summary_csv_output}")

    # Save condensed metrics
    metrics_with_f2 = []
    for split in ['train', 'val', 'test']:
        if split in summary_results:
            macro_metrics = summary_results[split]['macro_metrics']
            micro_metrics = summary_results[split]['micro_metrics']
            metrics_with_f2.append({
                'split': split,
                'metric_type': 'macro',
                'samples': macro_metrics['num_samples'],
                'positives': macro_metrics['num_positives'],
                'f1_score': macro_metrics['best_f1'],
                'f2_score': macro_metrics['best_f2'],
                'f5_score': macro_metrics['best_f5'],
                'f10_score': macro_metrics['best_f10'],
                'cohen_kappa': macro_metrics['best_cohen_kappa'],
                'precision': macro_metrics['best_precision'],
                'recall': macro_metrics['best_recall'],
                'auc': macro_metrics['auc'],
                'average_precision': macro_metrics['average_precision']
            })
            metrics_with_f2.append({
                'split': split,
                'metric_type': 'micro',
                'samples': micro_metrics['total_samples'],
                'positives': micro_metrics['total_positives'],
                'f1_score': micro_metrics['best_f1'],
                'f2_score': micro_metrics['best_f2'],
                'f5_score': micro_metrics.get('best_f5', 0),
                'f10_score': micro_metrics.get('best_f10', 0),
                'cohen_kappa': micro_metrics.get('best_cohen_kappa', 0),
                'precision': micro_metrics['best_precision'],
                'recall': micro_metrics['best_recall'],
                'auc': micro_metrics['auc'],
                'average_precision': micro_metrics['average_precision']
            })
    if metrics_with_f2:
        df_f2 = pd.DataFrame(metrics_with_f2)
        f2_output = os.path.join(save_dir, 'summary_metrics_with_f2.csv')
        df_f2.to_csv(f2_output, index=False)
        print(f"  Saved F2 metrics to {f2_output}")

    # Plots and report at overall level
    plot_confusion_matrices(summary_results, save_dir)
    plot_recall_by_identity(identity_results, save_dir)
    plot_f2_score_analysis(identity_results, summary_results, save_dir)
    generate_detailed_report(identity_results, summary_results, save_dir)

    # Save skipped evaluations info
    if skipped_evaluations and skipped_evaluations.get('skipped_entries'):
        skipped_output = os.path.join(save_dir, 'skipped_evaluations.json')
        with open(skipped_output, 'w') as f:
            json.dump(skipped_evaluations, f, indent=2)
        print(f"  Saved skipped evaluations info to {skipped_output}")

        skipped_csv_output = os.path.join(save_dir, 'skipped_evaluations.csv')
        skipped_rows = []
        for entry in skipped_evaluations['skipped_entries']:
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

    print(f"  All results saved to: {save_dir}")

    # Also write per top-N folders
    for N in [3, 5, 7]:
        topN_dir = os.path.join(save_dir, f'top{N}')
        os.makedirs(topN_dir, exist_ok=True)

        # Filter identity and summary for top-N (macro/micro recomputed)
        identity_results_topN = {}
        summary_results_topN = {}
        for split in ['train', 'val', 'test']:
            if split in identity_results and identity_results[split]:
                sorted_identities = sorted(identity_results[split].items(), key=lambda x: x[1]['best_f2'], reverse=True)
                topN_identities = sorted_identities[:N]
                identity_results_topN[split] = {k: v for k, v in topN_identities}
                # Save identity_results
                results_list = []
                for identity, metrics in identity_results_topN[split].items():
                    row = {'identity': identity, 'split': split}
                    row.update(metrics)
                    results_list.append(row)
                df = pd.DataFrame(results_list)
                output_file = os.path.join(topN_dir, f'identity_results_{split}_top{N}.csv')
                df.to_csv(output_file, index=False)
        # Build summary rows
        summary_rows_topN = []
        for split, split_results in summary_results_topN.items():
            for metric_type, metrics in split_results.items():
                if isinstance(metrics, dict):
                    row = {'split': split, 'metric_type': metric_type}
                    row.update(metrics)
                    summary_rows_topN.append(row)
        if summary_rows_topN:
            df_summary_topN = pd.DataFrame(summary_rows_topN)
            # Ensure std columns exist (optional; depends on prior computation)
            for metric in ['auc', 'average_precision', 'best_accuracy', 'best_precision', 'best_recall', 'best_f1', 'best_f2', 'best_f5', 'best_f10']:
                for nN in [3, 5, 7]:
                    col = f'{metric}@{nN}_std'
                    if col not in df_summary_topN.columns:
                        df_summary_topN[col] = None
            summary_csv_output_topN = os.path.join(topN_dir, f'summary_results_top{N}.csv')
            df_summary_topN.to_csv(summary_csv_output_topN, index=False)
            print(f"  Saved summary results for top{N} to {summary_csv_output_topN}")

        # Condensed top-N metrics
        metrics_with_f2_topN = []
        for split in ['train', 'val', 'test']:
            if split in summary_results_topN:
                macro_metrics = summary_results_topN[split]['macro_metrics']
                micro_metrics = summary_results_topN[split]['micro_metrics']
                metrics_with_f2_topN.append({
                    'split': split,
                    'metric_type': 'macro',
                    'samples': macro_metrics['num_samples'],
                    'positives': macro_metrics['num_positives'],
                    'f1_score': macro_metrics['best_f1'],
                    'f2_score': macro_metrics['best_f2'],
                    'f5_score': macro_metrics['best_f5'],
                    'f10_score': macro_metrics['best_f10'],
                    'cohen_kappa': macro_metrics['best_cohen_kappa'],
                    'precision': macro_metrics['best_precision'],
                    'recall': macro_metrics['best_recall'],
                    'auc': macro_metrics['auc'],
                    'average_precision': macro_metrics['average_precision']
                })
                metrics_with_f2_topN.append({
                    'split': split,
                    'metric_type': 'micro',
                    'samples': micro_metrics['total_samples'],
                    'positives': micro_metrics['total_positives'],
                    'f1_score': micro_metrics['best_f1'],
                    'f2_score': micro_metrics['best_f2'],
                    'f5_score': micro_metrics.get('best_f5', 0),
                    'f10_score': micro_metrics.get('best_f10', 0),
                    'cohen_kappa': micro_metrics.get('best_cohen_kappa', 0),
                    'precision': micro_metrics['best_precision'],
                    'recall': micro_metrics['best_recall'],
                    'auc': micro_metrics['auc'],
                    'average_precision': micro_metrics['average_precision']
                })
        if metrics_with_f2_topN:
            df_f2_topN = pd.DataFrame(metrics_with_f2_topN)
            f2_output_topN = os.path.join(topN_dir, f'summary_metrics_with_f2_top{N}.csv')
            df_f2_topN.to_csv(f2_output_topN, index=False)
            print(f"  Saved F2/F5/F10 metrics for top{N} to {f2_output_topN}")

        # Visualizations and reports for top-N
        plot_confusion_matrices(summary_results_topN, topN_dir)
        plot_recall_by_identity(identity_results_topN, topN_dir)
        plot_f2_score_analysis(identity_results_topN, summary_results_topN, topN_dir)
        generate_detailed_report(identity_results_topN, summary_results_topN, topN_dir)
        print(f"  All top{N} results saved to: {topN_dir}")


def print_final_summary_report(summary_results, valid_snippet_counts=None):
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)
    for split in ['train', 'val', 'test']:
        if split in summary_results:
            results = summary_results[split]
            print(f"\n{split.upper()} SPLIT:")
            print(f"  Number of identities: {results['num_identities']}")
            if valid_snippet_counts is not None:
                print(f"  Valid snippets included: {valid_snippet_counts.get(split, 0)}")
            micro = results['micro_metrics']
            macro = results['macro_metrics']
            topN = results.get('topN_metrics', {})
            print(f"  Micro Metrics:")
            print(f"    AUC: {micro['auc']:.4f} ± {micro.get('auc_std', 0):.4f}")
            print(f"    AP: {micro['average_precision']:.4f} ± {micro.get('average_precision_std', 0):.4f}")
            print(f"    Best F1: {micro['best_f1']:.4f} ± {micro.get('best_f1_std', 0):.4f}")
            print(f"    Best F2: {micro['best_f2']:.4f} ± {micro.get('best_f2_std', 0):.4f}")
            print(f"    Best F5: {micro.get('best_f5', 0):.4f}")
            print(f"    Best F10: {micro.get('best_f10', 0):.4f}")
            print(f"    Best Recall: {micro['best_recall']:.4f} ± {micro.get('best_recall_std', 0):.4f}")
            print(f"    Best Precision: {micro['best_precision']:.4f} ± {micro.get('best_precision_std', 0):.4f}")
            print(f"    Best Accuracy: {micro['best_accuracy']:.4f} ± {micro.get('best_accuracy_std', 0):.4f}")
            print(f"    Total Samples: {micro['total_samples']}")
            print(f"    Total Positives: {micro['total_positives']}")
            print(f"  Macro Metrics:")
            print(f"    AUC: {macro['auc']:.4f}")
            print(f"    AP: {macro['average_precision']:.4f}")
            print(f"    Best F1: {macro['best_f1']:.4f}")
            print(f"    Best F2: {macro['best_f2']:.4f}")
            print(f"    Best F5: {macro.get('best_f5', 0):.4f}")
            print(f"    Best F10: {macro.get('best_f10', 0):.4f}")
            if 'best_cohen_kappa' in macro:
                print(f"    Cohen Kappa: {macro.get('best_cohen_kappa', 0):.4f}")
            if topN:
                print(f"  Top-N Metrics (mean ± std over top N identities by F2):")
                for N in sorted(topN.keys()):
                    metrics = topN[N]
                    print(f"    Top-{N}:")
                    for metric in [
                        'auc', 'average_precision', 'best_accuracy', 'best_precision', 'best_recall',
                        'best_f1', 'best_f2', 'best_f5', 'best_f10', 'best_cohen_kappa']:
                        mean = metrics.get(f'{metric}@{N}', None)
                        std = metrics.get(f'{metric}@{N}_std', None)
                        if mean is not None and std is not None:
                            print(f"      {metric}: {mean:.4f} ± {std:.4f}")
