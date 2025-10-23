#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lab2 数据加载和验证模块
用于加载和验证LIBSVM格式的数据集
"""

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class Lab2DataLoader:
    """Lab2数据集加载器"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_breast_cancer_data(self, file_path="lab2/data/breast-cancer_scale"):
        """
        加载乳腺癌数据集（二分类）

        Returns:
            X_train, X_test, y_train, y_test: 训练和测试数据
        """
        print("📊 加载乳腺癌数据集...")

        # 加载LIBSVM格式数据
        X, y = load_svmlight_file(file_path)
        X = X.toarray()  # 转换为密集数组

        # 将标签转换为0,1（原来是2,4）
        y = np.where(y == 2, 0, 1)

        print(f"  数据形状: {X.shape}")
        print(f"  标签分布: {np.bincount(y)}")
        print(f"  特征数量: {X.shape[1]}")

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # 标准化特征
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        print(f"  训练集: {X_train.shape}")
        print(f"  测试集: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def load_iris_data(self, file_path="lab2/data/iris.scale"):
        """
        加载鸢尾花数据集（多分类）

        Returns:
            X_train, X_test, y_train, y_test: 训练和测试数据
        """
        print("📊 加载鸢尾花数据集...")

        # 加载LIBSVM格式数据
        X, y = load_svmlight_file(file_path)
        X = X.toarray()  # 转换为密集数组

        # 将标签转换为0,1,2（原来是1,2,3）
        y = y - 1

        print(f"  数据形状: {X.shape}")
        print(f"  标签分布: {np.bincount(y.astype(int))}")
        print(f"  特征数量: {X.shape[1]}")
        print(f"  类别数量: {len(np.unique(y))}")

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # 标准化特征
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        print(f"  训练集: {X_train.shape}")
        print(f"  测试集: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def visualize_data(self, X, y, dataset_name, feature_indices=(0, 1)):
        """
        可视化数据集

        Args:
            X: 特征矩阵
            y: 标签向量
            dataset_name: 数据集名称
            feature_indices: 要可视化的特征索引
        """
        plt.figure(figsize=(10, 8))

        # 选择两个特征进行可视化
        X_vis = X[:, feature_indices]

        # 散点图
        unique_labels = np.unique(y)
        colors = ['red', 'green', 'blue', 'orange', 'purple']

        for i, label in enumerate(unique_labels):
            mask = y == label
            plt.scatter(X_vis[mask, 0], X_vis[mask, 1],
                       c=colors[i % len(colors)], label=f'类别 {int(label)}',
                       alpha=0.7, s=50)

        plt.xlabel(f'特征 {feature_indices[0] + 1}')
        plt.ylabel(f'特征 {feature_indices[1] + 1}')
        plt.title(f'{dataset_name} 数据集散点图')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存图片
        output_path = f"lab2/outputs/{dataset_name.lower().replace(' ', '_')}_scatter.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"  📈 散点图已保存到: {output_path}")

    def plot_feature_distribution(self, X, y, dataset_name):
        """
        绘制特征分布图

        Args:
            X: 特征矩阵
            y: 标签向量
            dataset_name: 数据集名称
        """
        n_features = min(6, X.shape[1])  # 最多显示6个特征
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        unique_labels = np.unique(y)

        for i in range(n_features):
            ax = axes[i]

            for label in unique_labels:
                mask = y == label
                ax.hist(X[mask, i], bins=20, alpha=0.7,
                       label=f'类别 {int(label)}', density=True)

            ax.set_title(f'特征 {i+1} 分布')
            ax.set_xlabel('特征值')
            ax.set_ylabel('密度')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # 保存图片
        output_path = f"lab2/outputs/{dataset_name.lower().replace(' ', '_')}_features.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"  📊 特征分布图已保存到: {output_path}")

def test_data_loading():
    """测试数据加载功能"""
    print("🧪 测试Lab2数据加载功能")
    print("=" * 50)

    loader = Lab2DataLoader()

    # 测试乳腺癌数据集
    print("\n" + "="*20 + " 乳腺癌数据集测试 " + "="*20)
    X_train_bc, X_test_bc, y_train_bc, y_test_bc = loader.load_breast_cancer_data()

    # 可视化乳腺癌数据
    loader.visualize_data(X_train_bc, y_train_bc, "乳腺癌数据集", (0, 1))
    loader.plot_feature_distribution(X_train_bc, y_train_bc, "乳腺癌数据集")

    # 测试鸢尾花数据集
    print("\n" + "="*20 + " 鸢尾花数据集测试 " + "="*20)
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = loader.load_iris_data()

    # 可视化鸢尾花数据
    loader.visualize_data(X_train_iris, y_train_iris, "鸢尾花数据集", (0, 1))
    loader.plot_feature_distribution(X_train_iris, y_train_iris, "鸢尾花数据集")

    print("\n✅ 所有数据加载测试完成！")
    print("\n数据集总结:")
    print(f"  乳腺癌数据集: 训练集 {X_train_bc.shape}, 测试集 {X_test_bc.shape}")
    print(f"  鸢尾花数据集: 训练集 {X_train_iris.shape}, 测试集 {X_test_iris.shape}")

if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    test_data_loading()