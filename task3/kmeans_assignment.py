"""
kmeans_assignment.py
K-Means 聚类作业 - 按照 lab03.pdf 顺序组织

所有代码按照 PDF 章节顺序整合，每部分开始前有标题注释。
关键步骤包含中文注释说明。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
)
from time import time


# ============================================================================
# 1. K-means for the iris data
# ============================================================================
# 在本部分实现 K-means 算法并应用于 Iris 数据集

# 1.1 加载 Iris 数据集
print("\n" + "="*80)
print("1. K-means for the iris data")
print("="*80)

iris = load_iris()
# 选取特征索引 1 和 3 用于二维可视化（sepal width 和 petal width）
X = iris.data[:, [1, 3]]
y = iris.target  # 真实标签

print(f"Loaded Iris dataset: {X.shape[0]} samples, 2 features selected")
print(f"Features used: {iris.feature_names[1]}, {iris.feature_names[3]}")
print(f"Number of classes: {len(np.unique(y))}")


# 1.2 实现 K-means 算法类
class MyKMeans:
    """
    自定义 K-means 实现（Lloyd 算法）
    
    参数:
        n_clusters: 簇的数量
        max_iter: 最大迭代次数
        tol: 收敛阈值（质心移动距离小于此值时停止）
        random_state: 随机种子
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        """
        训练 K-means 模型
        
        步骤:
        1. 随机初始化 k 个质心
        2. 迭代：
           a) 分配：每个样本分配到最近的质心
           b) 更新：重新计算每个簇的质心
        3. 直到收敛或达到最大迭代次数
        """
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        # 步骤1：随机初始化质心 - 从样本中随机选择 k 个点作为初始质心
        initial_idx = rng.choice(n_samples, self.n_clusters, replace=False)
        centers = X[initial_idx].astype(float)

        # 步骤2：迭代优化
        for itr in range(self.max_iter):
            # 步骤2a：分配步骤 - 计算每个样本到所有质心的距离，分配到最近的簇
            # 使用广播计算距离矩阵: X[:, None, :] - centers[None, :, :]
            distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            # 步骤2b：更新步骤 - 计算每个簇的新质心（簇内样本的均值）
            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) == 0:
                    # 处理空簇：用随机样本重新初始化该质心
                    new_centers[k] = X[rng.randint(0, n_samples)]
                else:
                    new_centers[k] = members.mean(axis=0)

            # 步骤3：检查收敛条件 - 质心移动距离小于阈值
            center_shift = np.linalg.norm(centers - new_centers)
            centers = new_centers
            if center_shift <= self.tol:
                print(f"  Converged at iteration {itr + 1}")
                break

        # 保存最终结果
        self.cluster_centers_ = centers
        self.labels_ = labels
        # 计算 inertia（簇内平方和，衡量簇的紧密程度）
        self.inertia_ = np.sum((X - centers[labels]) ** 2)
        return self

    def predict(self, X):
        """预测新样本的簇标签"""
        distances = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return np.argmin(distances, axis=1)


# 1.3 在 Iris 数据集上训练自定义 K-means
print("\nTraining custom K-means on Iris dataset...")
my_kmeans = MyKMeans(n_clusters=3, random_state=42)
t0 = time()
my_kmeans.fit(X)
my_time = time() - t0
print(f"Training time: {my_time:.4f}s")
print(f"Inertia (within-cluster sum of squares): {my_kmeans.inertia_:.4f}")
print(f"Cluster centers:\n{my_kmeans.cluster_centers_}")


# ============================================================================
# 2. Plotting results
# ============================================================================
# 可视化聚类结果

print("\n" + "="*80)
print("2. Plotting results")
print("="*80)

# 创建输出目录
output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 绘制三个子图：真实标签、自定义 K-means 结果、sklearn K-means 结果
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 子图1：真实标签
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30, edgecolors='k', alpha=0.7)
axes[0].set_title('True Labels (Iris Species)', fontsize=12, fontweight='bold')
axes[0].set_xlabel(iris.feature_names[1], fontsize=10)
axes[0].set_ylabel(iris.feature_names[3], fontsize=10)
axes[0].grid(True, alpha=0.3)

# 子图2：自定义 K-means 结果
axes[1].scatter(X[:, 0], X[:, 1], c=my_kmeans.labels_, cmap='viridis', s=30, edgecolors='k', alpha=0.7)
axes[1].scatter(my_kmeans.cluster_centers_[:, 0], my_kmeans.cluster_centers_[:, 1], 
                c='red', marker='X', s=200, edgecolors='black', linewidths=2, label='Centroids')
axes[1].set_title('My K-means Results', fontsize=12, fontweight='bold')
axes[1].set_xlabel(iris.feature_names[1], fontsize=10)
axes[1].set_ylabel(iris.feature_names[3], fontsize=10)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 暂时留空第三个子图（将在第4部分填充 sklearn 结果）
axes[2].text(0.5, 0.5, 'Sklearn results\n(see section 4)', 
             ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
axes[2].set_xlabel(iris.feature_names[1], fontsize=10)
axes[2].set_ylabel(iris.feature_names[3], fontsize=10)

plt.tight_layout()
plot_file = os.path.join(output_dir, 'iris_clustering_results.png')
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"Saved plot: {plot_file}")
plt.close()


# ============================================================================
# 3. Measuring performance
# ============================================================================
# 使用多种指标评估聚类性能

print("\n" + "="*80)
print("3. Measuring performance")
print("="*80)

def evaluate_clustering(X, labels, true_labels=None):
    """
    计算聚类评估指标
    
    指标说明:
    - Silhouette Score: 衡量样本与其簇内其他样本的相似度 vs 与其他簇的相似度
      范围 [-1, 1]，越接近 1 表示聚类质量越好
    - Adjusted Rand Index (ARI): 衡量聚类结果与真实标签的一致性
      范围 [-1, 1]，1 表示完全一致，0 表示随机分配
    - Homogeneity: 每个簇是否只包含单一类别的样本
      范围 [0, 1]，1 表示完美的同质性
    - Completeness: 同一类别的样本是否被分配到同一个簇
      范围 [0, 1]，1 表示完美的完整性
    """
    results = {}
    
    # Silhouette Score（轮廓系数）
    try:
        results['silhouette_score'] = silhouette_score(X, labels)
    except Exception as e:
        results['silhouette_score'] = np.nan
        print(f"  Warning: Could not compute silhouette score - {e}")

    # 如果有真实标签，计算外部评估指标
    if true_labels is not None:
        results['adjusted_rand_index'] = adjusted_rand_score(true_labels, labels)
        results['homogeneity'] = homogeneity_score(true_labels, labels)
        results['completeness'] = completeness_score(true_labels, labels)
    
    return results


# 评估自定义 K-means 的性能
print("\nPerformance metrics for custom K-means:")
my_metrics = evaluate_clustering(X, my_kmeans.labels_, y)
for metric_name, metric_value in my_metrics.items():
    print(f"  {metric_name}: {metric_value:.4f}")


# ============================================================================
# 4. Comparing with scikit-learn
# ============================================================================
# 与 scikit-learn 的 K-means 实现进行对比

print("\n" + "="*80)
print("4. Comparing with scikit-learn")
print("="*80)

# 训练 sklearn K-means
print("\nTraining sklearn K-means...")
sklearn_kmeans = SKLearnKMeans(n_clusters=3, random_state=42, n_init=10)
t0 = time()
sklearn_kmeans.fit(X)
sklearn_time = time() - t0

print(f"Training time: {sklearn_time:.4f}s")
print(f"Inertia: {sklearn_kmeans.inertia_:.4f}")
print(f"Cluster centers:\n{sklearn_kmeans.cluster_centers_}")

# 评估 sklearn K-means 的性能
print("\nPerformance metrics for sklearn K-means:")
sklearn_metrics = evaluate_clustering(X, sklearn_kmeans.labels_, y)
for metric_name, metric_value in sklearn_metrics.items():
    print(f"  {metric_name}: {metric_value:.4f}")

# 对比两种实现
print("\n" + "-"*80)
print("Comparison Summary:")
print("-"*80)
print(f"{'Metric':<30} {'My K-means':<20} {'Sklearn K-means':<20}")
print("-"*80)
print(f"{'Training time (s)':<30} {my_time:<20.4f} {sklearn_time:<20.4f}")
print(f"{'Inertia':<30} {my_kmeans.inertia_:<20.4f} {sklearn_kmeans.inertia_:<20.4f}")
for metric_name in my_metrics.keys():
    my_val = my_metrics[metric_name]
    sk_val = sklearn_metrics[metric_name]
    print(f"{metric_name:<30} {my_val:<20.4f} {sk_val:<20.4f}")
print("-"*80)

# 绘制完整的对比图（更新之前的图）
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 子图1：真实标签
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30, edgecolors='k', alpha=0.7)
axes[0].set_title('True Labels', fontsize=12, fontweight='bold')
axes[0].set_xlabel(iris.feature_names[1], fontsize=10)
axes[0].set_ylabel(iris.feature_names[3], fontsize=10)
axes[0].grid(True, alpha=0.3)

# 子图2：自定义 K-means
axes[1].scatter(X[:, 0], X[:, 1], c=my_kmeans.labels_, cmap='viridis', s=30, edgecolors='k', alpha=0.7)
axes[1].scatter(my_kmeans.cluster_centers_[:, 0], my_kmeans.cluster_centers_[:, 1], 
                c='red', marker='X', s=200, edgecolors='black', linewidths=2)
axes[1].set_title(f'My K-means (inertia={my_kmeans.inertia_:.2f})', fontsize=12, fontweight='bold')
axes[1].set_xlabel(iris.feature_names[1], fontsize=10)
axes[1].set_ylabel(iris.feature_names[3], fontsize=10)
axes[1].grid(True, alpha=0.3)

# 子图3：sklearn K-means
axes[2].scatter(X[:, 0], X[:, 1], c=sklearn_kmeans.labels_, cmap='viridis', s=30, edgecolors='k', alpha=0.7)
axes[2].scatter(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1], 
                c='red', marker='X', s=200, edgecolors='black', linewidths=2)
axes[2].set_title(f'Sklearn K-means (inertia={sklearn_kmeans.inertia_:.2f})', fontsize=12, fontweight='bold')
axes[2].set_xlabel(iris.feature_names[1], fontsize=10)
axes[2].set_ylabel(iris.feature_names[3], fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
comparison_file = os.path.join(output_dir, 'iris_comparison_my_vs_sklearn.png')
plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
print(f"\nSaved comparison plot: {comparison_file}")
plt.close()


# ============================================================================
# 5. More
# ============================================================================
# 使用 make_blobs 生成二维数据集进行额外实验

print("\n" + "="*80)
print("5. More - Experiments with make_blobs (2D data)")
print("="*80)

# 生成多个不同特性的数据集进行测试
datasets_config = [
    {'n_samples': 300, 'centers': 3, 'cluster_std': 1.0, 'random_state': 10, 'name': 'Easy (3 clusters, std=1.0)'},
    {'n_samples': 300, 'centers': 4, 'cluster_std': 0.5, 'random_state': 20, 'name': 'Easy (4 clusters, std=0.5)'},
    {'n_samples': 300, 'centers': 3, 'cluster_std': 2.0, 'random_state': 30, 'name': 'Hard (3 clusters, std=2.0)'},
]

for idx, config in enumerate(datasets_config, start=1):
    print(f"\n{'-'*80}")
    print(f"Dataset {idx}: {config['name']}")
    print(f"{'-'*80}")
    
    # 生成数据（特征维度统一设置为 2）
    X_blob, y_blob = make_blobs(
        n_samples=config['n_samples'], 
        centers=config['centers'], 
        n_features=2,  # 按要求设置为 2 维
        cluster_std=config['cluster_std'], 
        random_state=config['random_state']
    )
    
    print(f"Generated {X_blob.shape[0]} samples with {X_blob.shape[1]} features")
    print(f"True number of clusters: {config['centers']}")
    
    # 使用自定义 K-means
    print("\n[My K-means]")
    my_km_blob = MyKMeans(n_clusters=config['centers'], random_state=42)
    t0 = time()
    my_km_blob.fit(X_blob)
    my_time_blob = time() - t0
    my_metrics_blob = evaluate_clustering(X_blob, my_km_blob.labels_, y_blob)
    
    print(f"  Time: {my_time_blob:.4f}s, Inertia: {my_km_blob.inertia_:.4f}")
    for k, v in my_metrics_blob.items():
        print(f"  {k}: {v:.4f}")
    
    # 使用 sklearn K-means
    print("\n[Sklearn K-means]")
    sk_km_blob = SKLearnKMeans(n_clusters=config['centers'], random_state=42, n_init=10)
    t0 = time()
    sk_km_blob.fit(X_blob)
    sk_time_blob = time() - t0
    sk_metrics_blob = evaluate_clustering(X_blob, sk_km_blob.labels_, y_blob)
    
    print(f"  Time: {sk_time_blob:.4f}s, Inertia: {sk_km_blob.inertia_:.4f}")
    for k, v in sk_metrics_blob.items():
        print(f"  {k}: {v:.4f}")
    
    # 按要求：在每个算法完成后输出一张算法结果和标签结果的对比图
    # 为自定义 K-means 生成对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].scatter(X_blob[:, 0], X_blob[:, 1], c=my_km_blob.labels_, cmap='tab10', s=30, edgecolors='k', alpha=0.7)
    axes[0].scatter(my_km_blob.cluster_centers_[:, 0], my_km_blob.cluster_centers_[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    axes[0].set_title(f'My K-means Results\n(ARI={my_metrics_blob["adjusted_rand_index"]:.3f})', 
                     fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap='tab10', s=30, edgecolors='k', alpha=0.7)
    axes[1].set_title('True Labels', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    my_blob_file = os.path.join(output_dir, f'more_dataset{idx}_my_kmeans_vs_true.png')
    plt.savefig(my_blob_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {my_blob_file}")
    plt.close()
    
    # 为 sklearn K-means 生成对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].scatter(X_blob[:, 0], X_blob[:, 1], c=sk_km_blob.labels_, cmap='tab10', s=30, edgecolors='k', alpha=0.7)
    axes[0].scatter(sk_km_blob.cluster_centers_[:, 0], sk_km_blob.cluster_centers_[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    axes[0].set_title(f'Sklearn K-means Results\n(ARI={sk_metrics_blob["adjusted_rand_index"]:.3f})', 
                     fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap='tab10', s=30, edgecolors='k', alpha=0.7)
    axes[1].set_title('True Labels', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    sk_blob_file = os.path.join(output_dir, f'more_dataset{idx}_sklearn_kmeans_vs_true.png')
    plt.savefig(sk_blob_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {sk_blob_file}")
    plt.close()


# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("All tasks completed!")
print("="*80)
print(f"All figures have been saved to: {output_dir}")
print("\nGenerated files:")
for filename in sorted(os.listdir(output_dir)):
    if filename.endswith('.png'):
        print(f"  - {filename}")
print("\nMetrics for all experiments have been printed above.")
print("="*80)
