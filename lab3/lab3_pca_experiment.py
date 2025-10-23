#!/usr/bin/env python3
"""
Lab3: PCA降维与可视化实验
MNIST手写数字数据集降维实践

实验内容：
1. 加载MNIST数据集，选取每类100个样本
2. 数据预处理（展平、标准化）
3. PCA降维与方差分析
4. 降维可视化与分类性能对比
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_mnist_subset(samples_per_class=100):
    """
    加载MNIST数据集并选取每类指定数量的样本

    Args:
        samples_per_class: 每个类别的样本数量

    Returns:
        X: 特征矩阵 (n_samples, 784)
        y: 标签向量 (n_samples,)
    """
    print("🔄 正在加载MNIST数据集...")

    # 从OpenML加载MNIST数据集
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target

    # 转换标签为整数
    y = y.astype(int)

    print(f"📊 原始数据集大小: {X.shape[0]} 个样本, {X.shape[1]} 个特征")

    # 选取每类指定数量的样本
    X_subset = []
    y_subset = []

    for digit in range(10):
        # 找到当前数字的所有索引
        indices = np.where(y == digit)[0]
        # 随机选取指定数量的样本
        selected_indices = np.random.choice(indices, samples_per_class, replace=False)
        X_subset.append(X[selected_indices])
        y_subset.append(y[selected_indices])

    # 合并所有类别的样本
    X_subset = np.vstack(X_subset)
    y_subset = np.hstack(y_subset)

    print(f"✅ 选取后的数据集大小: {X_subset.shape[0]} 个样本")
    print(f"📈 每个类别样本数: {samples_per_class}")

    return X_subset, y_subset

def preprocess_data(X, y):
    """
    数据预处理：展平和标准化

    Args:
        X: 特征矩阵
        y: 标签向量

    Returns:
        X_scaled: 标准化后的特征矩阵
        y: 标签向量
    """
    print("\n🔄 正在进行数据预处理...")

    # 数据已经是展平的784维向量，只需要标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"✅ 数据预处理完成")
    print(f"📊 特征维度: {X_scaled.shape[1]}")
    print(f"📈 数据均值: {np.mean(X_scaled):.6f} (应接近0)")
    print(f"📈 数据标准差: {np.std(X_scaled):.6f} (应接近1)")

    return X_scaled, y

def perform_pca_analysis(X, n_components=200):
    """
    执行PCA降维分析

    Args:
        X: 标准化后的特征矩阵
        n_components: 要保留的主成分数量

    Returns:
        pca: PCA模型对象
        X_pca: 降维后的数据
        explained_variance_ratio: 方差贡献率
        cumulative_variance_ratio: 累计方差贡献率
    """
    print(f"\n🔄 正在进行PCA降维分析（保留{n_components}个主成分）...")

    # 创建PCA模型，保留更多主成分以便找到95%方差阈值
    pca = PCA(n_components=n_components)

    # 拟合模型并转换数据
    X_pca = pca.fit_transform(X)

    # 获取方差贡献率
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # 找到达到95%方差的主成分数量
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1

    print(f"✅ PCA降维完成")
    print(f"📊 原始维度: {X.shape[1]}")
    print(f"📊 降维后维度: {X_pca.shape[1]}")
    print(f"📈 前{n_components}个主成分总方差保留率: {cumulative_variance_ratio[-1]:.4f} ({cumulative_variance_ratio[-1]*100:.2f}%)")
    print(f"📈 达到95%方差需要的主成分数: {n_components_95}")

    return pca, X_pca, explained_variance_ratio, cumulative_variance_ratio, n_components_95

def plot_variance_analysis(cumulative_variance_ratio, save_path=None):
    """
    绘制累计方差贡献率曲线

    Args:
        cumulative_variance_ratio: 累计方差贡献率
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))

    # 绘制累计方差贡献率曲线
    plt.plot(range(1, len(cumulative_variance_ratio) + 1),
             cumulative_variance_ratio * 100,
             'b-', linewidth=2, marker='o', markersize=4)

    # 添加95%方差线
    plt.axhline(y=95, color='r', linestyle='--', alpha=0.7, label='95% 方差阈值')

    # 找到达到95%方差的主成分数量
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    plt.axvline(x=n_components_95, color='r', linestyle='--', alpha=0.7)

    plt.xlabel('主成分数量', fontsize=12)
    plt.ylabel('累计方差贡献率 (%)', fontsize=12)
    plt.title('PCA累计方差贡献率曲线', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 添加注释
    plt.annotate(f'前{n_components_95}个主成分\n达到95%方差',
                 xy=(n_components_95, 95),
                 xytext=(n_components_95 + 2, 85),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 方差分析图已保存: {save_path}")

    plt.show()

    return n_components_95

def plot_2d_pca_visualization(X_pca, y, save_path=None):
    """
    绘制2D PCA降维可视化散点图

    Args:
        X_pca: 降维后的数据（至少2维）
        y: 标签向量
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 8))

    # 定义颜色
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # 为每个数字类别绘制散点图
    for digit in range(10):
        mask = y == digit
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors[digit]], label=f'数字 {digit}',
                   alpha=0.7, s=50, edgecolors='black', linewidth=0.5)

    plt.xlabel('第一主成分', fontsize=12)
    plt.ylabel('第二主成分', fontsize=12)
    plt.title('MNIST数据集PCA降维2D可视化', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 2D可视化图已保存: {save_path}")

    plt.show()

def train_svm_classifier(X_train, X_test, y_train, y_test, kernel='rbf'):
    """
    训练SVM分类器

    Args:
        X_train, X_test: 训练和测试特征
        y_train, y_test: 训练和测试标签
        kernel: SVM核函数类型

    Returns:
        accuracy: 测试集准确率
        training_time: 训练时间
    """
    print(f"🔄 正在训练SVM分类器 (kernel={kernel})...")

    # 创建SVM分类器
    svm = SVC(kernel=kernel, random_state=42)

    # 记录训练时间
    start_time = time.time()
    svm.fit(X_train, y_train)
    training_time = time.time() - start_time

    # 预测并计算准确率
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ SVM训练完成")
    print(f"📊 测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"⏱️  训练时间: {training_time:.4f} 秒")

    return accuracy, training_time

def compare_classification_performance(X_original, X_pca, y, n_components_95):
    """
    对比原始数据和降维数据的分类性能

    Args:
        X_original: 原始特征矩阵
        X_pca: PCA降维后的特征矩阵
        y: 标签向量
        n_components_95: 达到95%方差的主成分数量
    """
    print("\n" + "="*60)
    print("🔍 分类性能对比分析")
    print("="*60)

    # 划分训练集和测试集
    X_orig_train, X_orig_test, y_train, y_test = train_test_split(
        X_original, y, test_size=0.3, random_state=42, stratify=y)

    X_pca_train, X_pca_test, _, _ = train_test_split(
        X_pca, y, test_size=0.3, random_state=42, stratify=y)

    # 使用保留95%方差的主成分数量
    X_pca_95_train = X_pca_train[:, :n_components_95]
    X_pca_95_test = X_pca_test[:, :n_components_95]

    print(f"📊 训练集大小: {len(y_train)} 样本")
    print(f"📊 测试集大小: {len(y_test)} 样本")
    print(f"📊 原始特征维度: {X_orig_train.shape[1]}")
    print(f"📊 降维特征维度: {X_pca_95_train.shape[1]}")

    # 在原始数据上训练SVM
    print(f"\n🔹 原始数据 (784维) SVM训练:")
    orig_accuracy, orig_time = train_svm_classifier(
        X_orig_train, X_orig_test, y_train, y_test)

    # 在降维数据上训练SVM
    print(f"\n🔹 降维数据 ({n_components_95}维) SVM训练:")
    pca_accuracy, pca_time = train_svm_classifier(
        X_pca_95_train, X_pca_95_test, y_train, y_test)

    # 性能对比总结
    print(f"\n" + "="*60)
    print("📈 性能对比总结")
    print("="*60)
    print(f"原始数据 (784维):")
    print(f"  准确率: {orig_accuracy:.4f} ({orig_accuracy*100:.2f}%)")
    print(f"  训练时间: {orig_time:.4f} 秒")

    print(f"\n降维数据 ({n_components_95}维):")
    print(f"  准确率: {pca_accuracy:.4f} ({pca_accuracy*100:.2f}%)")
    print(f"  训练时间: {pca_time:.4f} 秒")

    print(f"\n🔍 性能变化:")
    print(f"  准确率变化: {pca_accuracy - orig_accuracy:+.4f} ({(pca_accuracy - orig_accuracy)*100:+.2f}%)")
    print(f"  训练时间变化: {pca_time - orig_time:+.4f} 秒")
    print(f"  训练速度提升: {orig_time/pca_time:.2f}x")
    print(f"  维度压缩率: {(1 - n_components_95/784)*100:.1f}%")

    return {
        'original_accuracy': orig_accuracy,
        'original_time': orig_time,
        'pca_accuracy': pca_accuracy,
        'pca_time': pca_time,
        'n_components_95': n_components_95
    }

def main():
    """主实验流程"""
    print("🚀 开始Lab3: PCA降维与可视化实验")
    print("="*60)

    # 设置随机种子以确保可重复性
    np.random.seed(42)

    # 1. 加载MNIST数据集
    X, y = load_mnist_subset(samples_per_class=100)

    # 2. 数据预处理
    X_scaled, y = preprocess_data(X, y)

    # 3. PCA降维分析（保留200个主成分以找到95%方差阈值）
    pca, X_pca, explained_variance_ratio, cumulative_variance_ratio, n_components_95 = perform_pca_analysis(X_scaled, n_components=200)

    # 4. 绘制方差分析图
    plot_variance_analysis(cumulative_variance_ratio, 'lab3/variance_analysis.png')

    # 5. 绘制2D可视化图
    plot_2d_pca_visualization(X_pca, y, 'lab3/pca_2d_visualization.png')

    # 6. 分类性能对比
    performance_results = compare_classification_performance(X_scaled, X_pca, y, n_components_95)

    print(f"\n🎉 Lab3实验完成！")
    print(f"📁 结果已保存到 lab3/ 目录")

if __name__ == "__main__":
    main()