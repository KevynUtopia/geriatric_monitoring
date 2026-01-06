import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_file = "/Users/kevynzhang/Downloads/st_evaluation/adapted_set_a/summary.csv"

# read the csv file and read precision and recall columns
data = pd.read_csv(csv_file)

print("数据基本信息:")
print(f"总图片数量: {len(data)}")
print(f"列名: {list(data.columns)}")
print("\n前5行数据:")
print(data[['precision', 'recall']].head())

# 检查数据质量
print(f"\n数据统计:")
print(f"Precision范围: {data['precision'].min():.4f} - {data['precision'].max():.4f}")
print(f"Recall范围: {data['recall'].min():.4f} - {data['recall'].max():.4f}")
print(f"Precision平均值: {data['precision'].mean():.4f}")
print(f"Recall平均值: {data['recall'].mean():.4f}")

# 检查是否有无效数据
print(f"\n无效数据检查:")
print(f"Precision中的NaN值: {data['precision'].isna().sum()}")
print(f"Recall中的NaN值: {data['recall'].isna().sum()}")
print(f"Precision中的无穷值: {np.isinf(data['precision']).sum()}")
print(f"Recall中的无穷值: {np.isinf(data['recall']).sum()}")

# 清理数据
data_clean = data.dropna(subset=['precision', 'recall'])
data_clean = data_clean[~np.isinf(data_clean['precision'])]
data_clean = data_clean[~np.isinf(data_clean['recall'])]

print(f"\n清理后数据数量: {len(data_clean)}")

if len(data_clean) == 0:
    print("错误: 清理后没有有效数据!")
    exit()

precision = data_clean['precision'].values
recall = data_clean['recall'].values

def analyze_data_distribution(precision, recall):
    """
    分析precision和recall的分布
    """
    print(f"\n数据分布分析:")
    print(f"Precision值分布:")
    print(f"  = 1.0: {(precision == 1.0).sum()} ({100*(precision == 1.0).sum()/len(precision):.1f}%)")
    print(f"  = 0.0: {(precision == 0.0).sum()} ({100*(precision == 0.0).sum()/len(precision):.1f}%)")
    print(f"  其他值: {((precision != 1.0) & (precision != 0.0)).sum()} ({100*((precision != 1.0) & (precision != 0.0)).sum()/len(precision):.1f}%)")
    
    print(f"\nRecall值分布:")
    print(f"  = 1.0: {(recall == 1.0).sum()} ({100*(recall == 1.0).sum()/len(recall):.1f}%)")
    print(f"  = 0.0: {(recall == 0.0).sum()} ({100*(recall == 0.0).sum()/len(recall):.1f}%)")
    print(f"  其他值: {((recall != 1.0) & (recall != 0.0)).sum()} ({100*((recall != 1.0) & (recall != 0.0)).sum()/len(recall):.1f}%)")

def voc_ap_11_point_interpolation(recall, precision):
    """
    VOC 2007 11点插值法计算AP值
    Args:
        recall: recall值列表
        precision: precision值列表
    Returns:
        ap: Average Precision值
        interpolated_precision: 插值后的precision值
    """
    # VOC 2007的11个固定recall点
    voc_recall_points = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # 将recall和precision转换为numpy数组并排序
    recall = np.array(recall)
    precision = np.array(precision)
    
    # 按recall排序（从大到小）
    sorted_indices = np.argsort(recall)[::-1]
    recall = recall[sorted_indices]
    precision = precision[sorted_indices]
    
    # 11点插值
    interpolated_precision = []
    
    for voc_recall in voc_recall_points:
        # 找到所有recall >= voc_recall的点的最大precision
        mask = recall >= voc_recall
        if np.any(mask):
            max_precision = np.max(precision[mask])
        else:
            max_precision = 0.0
        interpolated_precision.append(max_precision)
    
    # 计算AP（11个点的precision平均值）
    ap = np.mean(interpolated_precision)
    
    return ap, interpolated_precision

def plot_pr_curve(recall, precision, ap, interpolated_precision=None):
    """
    绘制PR曲线
    Args:
        recall: recall值
        precision: precision值
        ap: Average Precision值
        interpolated_precision: 插值后的precision值（可选）
    """
    plt.figure(figsize=(12, 10))
    
    # 创建子图
    plt.subplot(2, 2, 1)
    plt.scatter(recall, precision, alpha=0.6, s=30)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall Scatter Plot')
    plt.grid(True, alpha=0.3)
    
    # 绘制原始PR曲线（排序后）
    plt.subplot(2, 2, 2)
    sorted_indices = np.argsort(recall)[::-1]
    recall_sorted = recall[sorted_indices]
    precision_sorted = precision[sorted_indices]
    plt.plot(recall_sorted, precision_sorted, 'b-', linewidth=2, label=f'Original PR Curve (AP={ap:.3f})')
    
    # 如果提供了插值后的precision，绘制11点插值曲线
    if interpolated_precision is not None:
        voc_recall_points = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.plot(voc_recall_points, interpolated_precision, 'r--', linewidth=2, 
                label='VOC 2007 11-point Interpolation')
        plt.scatter(voc_recall_points, interpolated_precision, c='red', s=50, zorder=5)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve with VOC 2007 11-point Interpolation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    
    # 添加AP值文本
    plt.text(0.05, 0.95, f'AP = {ap:.3f}', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 数据分布直方图
    plt.subplot(2, 2, 3)
    plt.hist(precision, bins=20, alpha=0.7, color='blue', label='Precision')
    plt.xlabel('Precision')
    plt.ylabel('Frequency')
    plt.title('Precision Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.hist(recall, bins=20, alpha=0.7, color='green', label='Recall')
    plt.xlabel('Recall')
    plt.ylabel('Frequency')
    plt.title('Recall Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 分析数据分布
analyze_data_distribution(precision, recall)

# 计算AP值
ap, interpolated_precision = voc_ap_11_point_interpolation(recall, precision)

print(f"\nAverage Precision (AP) using VOC 2007 11-point interpolation: {ap:.4f}")
print(f"Interpolated precision values at 11 recall points:")
for i, (r, p) in enumerate(zip([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], interpolated_precision)):
    print(f"  Recall={r:.1f}: Precision={p:.4f}")

# 绘制PR曲线
plot_pr_curve(recall, precision, ap, interpolated_precision)
