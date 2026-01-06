import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrices(summary_results, save_dir):
    print("Generating confusion matrix plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    splits = ['train', 'val', 'test']
    for idx, split in enumerate(splits):
        ax = axes[idx]
        if split in summary_results:
            metrics = summary_results[split]['macro_metrics']
            cm = np.array([
                [metrics['true_negatives'], metrics['false_positives']],
                [metrics['false_negatives'], metrics['true_positives']]
            ])
            sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', ax=ax,
                       xticklabels=['Predicted Negative', 'Predicted Positive'],
                       yticklabels=['Actual Negative', 'Actual Positive'])
            f1_score = metrics['best_f1']
            f2_score = metrics['best_f2']
            precision = metrics['best_precision']
            recall = metrics['best_recall']
            auc = metrics['auc']
            ax.set_title(f'{split.upper()} Split\n'
                       f'F1: {f1_score:.3f} | F2: {f2_score:.3f} | Precision: {precision:.3f}\n'
                       f'Recall: {recall:.3f} | AUC: {auc:.3f}')
        else:
            ax.text(0.5, 0.5, f'No data for {split}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{split.upper()} Split')
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to {plot_path}")
    plt.close()

def plot_recall_by_identity(identity_results, save_dir):
    print("Generating recall by identity plots...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    splits = ['train', 'val', 'test']
    for idx, split in enumerate(splits):
        ax = axes[idx]
        if split in identity_results and identity_results[split]:
            identity_data = identity_results[split]
            identities = sorted(identity_data.keys(), key=lambda x: int(x.split('_')[1]))
            recalls = [identity_data[identity]['best_recall'] for identity in identities]
            bars = ax.bar(range(len(identities)), recalls, alpha=0.7)
            ax.set_xlabel('Identity')
            ax.set_ylabel('Recall')
            ax.set_title(f'{split.upper()} Split - Recall by Identity')
            ax.set_xticks(range(len(identities)))
            ax.set_xticklabels(identities, rotation=45)
            ax.set_ylim(0, 1)
            for bar, recall in zip(bars, recalls):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{recall:.3f}', ha='center', va='bottom', fontsize=8)
            avg_recall = np.mean(recalls)
            ax.axhline(y=avg_recall, color='red', linestyle='--', 
                      label=f'Average: {avg_recall:.3f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, f'No data for {split}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{split.upper()} Split')
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'recall_by_identity.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Recall by identity plot saved to {plot_path}")
    plt.close()

def plot_f2_score_analysis(identity_results, summary_results, save_dir):
    print("Generating F2 score analysis plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax1 = axes[0, 0]
    splits = ['train', 'val', 'test']
    f2_scores = []
    for split in splits:
        if split in summary_results:
            f2_scores.append(summary_results[split]['macro_metrics']['best_f2'])
        else:
            f2_scores.append(0)
    bars = ax1.bar(splits, f2_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax1.set_title('F2 Score by Split (Macro Average)')
    ax1.set_ylabel('F2 Score')
    ax1.set_ylim(0, 1)
    for bar, score in zip(bars, f2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    ax2 = axes[0, 1]
    all_f2_scores = []
    split_labels = []
    for split in splits:
        if split in identity_results and identity_results[split]:
            identity_data = identity_results[split]
            split_f2_scores = [identity_data[identity]['best_f2'] for identity in identity_data.keys()]
            all_f2_scores.extend(split_f2_scores)
            split_labels.extend([split] * len(split_f2_scores))
    if all_f2_scores:
        import pandas as pd
        df_f2 = pd.DataFrame({'F2_Score': all_f2_scores, 'Split': split_labels})
        for i, split in enumerate(splits):
            split_data = df_f2[df_f2['Split'] == split]['F2_Score']
            if len(split_data) > 0:
                ax2.violinplot([split_data], positions=[i], showmeans=True)
        ax2.set_xticks(range(len(splits)))
        ax2.set_xticklabels(splits)
        ax2.set_title('F2 Score Distribution by Split')
        ax2.set_ylabel('F2 Score')
    ax3 = axes[1, 0]
    all_identity_f2 = []
    for split in splits:
        if split in identity_results and identity_results[split]:
            identity_data = identity_results[split]
            for identity, metrics in identity_data.items():
                all_identity_f2.append((f"{identity}_{split}", metrics['best_f2']))
    if all_identity_f2:
        top_identities = sorted(all_identity_f2, key=lambda x: x[1], reverse=True)[:10]
        names = [x[0] for x in top_identities]
        scores = [x[1] for x in top_identities]
        bars = ax3.barh(range(len(names)), scores)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names)
        ax3.set_xlabel('F2 Score')
        ax3.set_title('Top 10 Identities by F2 Score')
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{score:.3f}', ha='left', va='center')
    ax4 = axes[1, 1]
    precisions = []
    recalls = []
    f2_scores = []
    colors = []
    color_map = {'train': 'blue', 'val': 'green', 'test': 'red'}
    for split in splits:
        if split in identity_results and identity_results[split]:
            identity_data = identity_results[split]
            for identity, metrics in identity_data.items():
                precisions.append(metrics['best_precision'])
                recalls.append(metrics['best_recall'])
                f2_scores.append(metrics['best_f2'])
                colors.append(color_map[split])
    if precisions and recalls:
        scatter = ax4.scatter(precisions, recalls, c=colors, alpha=0.6, s=50)
        ax4.set_xlabel('Precision')
        ax4.set_ylabel('Recall')
        ax4.set_title('Precision vs Recall (F2 Score Analysis)')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        precision_range = np.linspace(0.01, 1, 100)
        for f2_value in [0.1, 0.3, 0.5, 0.7, 0.9]:
            recall_line = (f2_value * precision_range) / (4 * precision_range + f2_value)
            ax4.plot(precision_range, recall_line, '--', alpha=0.3, 
                    label=f'F2={f2_value}')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[split], label=split) 
                          for split in splits]
        ax4.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'f2_score_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"F2 score analysis plot saved to {plot_path}")
    plt.close() 