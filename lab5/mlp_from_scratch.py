#!/usr/bin/env python3
"""
Lab5 Part A: MLP从零实现
多层感知机 (Multi-Layer Perceptron) 完全从零实现

功能:
- MNIST数据加载与预处理
- 从零实现神经网络所有组件
- 完整的训练与评估流程
- 结果可视化与性能分析

目标: 测试集准确率 > 90%
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# 设置中文字体和绘图参数
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

class MLPFromScratch:
    """
    从零实现的多层感知机类
    实现了完整的前向传播、反向传播和训练流程
    """

    def __init__(self, layer_sizes, learning_rate=0.01, random_seed=42):
        """
        初始化MLP网络

        Args:
            layer_sizes: 网络层结构列表 [输入层, 隐藏层1, ..., 输出层]
            learning_rate: 学习率
            random_seed: 随机种子
        """
        np.random.seed(random_seed)
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)

        # 初始化参数
        self.parameters = self.initialize_parameters()

        # 训练历史记录
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def initialize_parameters(self):
        """
        初始化网络参数 (权重和偏置)
        使用He初始化策略
        """
        parameters = {}

        for l in range(1, self.num_layers):
            # He初始化: W ~ N(0, sqrt(2/n[l-1]))
            parameters[f'W{l}'] = np.random.randn(
                self.layer_sizes[l], self.layer_sizes[l-1]
            ) * np.sqrt(2.0 / self.layer_sizes[l-1])

            # 偏置初始化为小随机数
            parameters[f'b{l}'] = np.random.randn(
                self.layer_sizes[l], 1
            ) * 0.01

        return parameters

    def relu(self, Z):
        """ReLU激活函数"""
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        """ReLU激活函数的导数"""
        return (Z > 0).astype(float)

    def softmax(self, Z):
        """
        Softmax激活函数
        实现数值稳定的版本
        """
        # 数值稳定性: 减去最大值
        Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def one_hot_encode(self, y, num_classes):
        """
        将标签向量进行独热编码

        Args:
            y: 标签向量 (shape: (m,))
            num_classes: 类别数量

        Returns:
            Y_one_hot: 独热编码矩阵 (shape: (num_classes, m))
        """
        m = y.shape[0]
        Y_one_hot = np.zeros((num_classes, m))
        Y_one_hot[y, np.arange(m)] = 1
        return Y_one_hot

    def forward_propagation(self, X):
        """
        前向传播

        Args:
            X: 输入数据 (shape: (n_features, m))

        Returns:
            cache: 缓存各层结果用于反向传播
        """
        cache = {'A0': X}
        A = X

        # 前向传播各层
        for l in range(1, self.num_layers - 1):
            Z = np.dot(self.parameters[f'W{l}'], A) + self.parameters[f'b{l}']
            A = self.relu(Z)
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A

        # 输出层使用softmax
        Z_L = np.dot(self.parameters[f'W{self.num_layers-1}'], A) + \
              self.parameters[f'b{self.num_layers-1}']
        A_L = self.softmax(Z_L)
        cache[f'Z{self.num_layers-1}'] = Z_L
        cache[f'A{self.num_layers-1}'] = A_L

        return cache

    def compute_loss(self, Y_true, Y_pred):
        """
        计算交叉熵损失

        Args:
            Y_true: 真实标签 (shape: (num_classes, m))
            Y_pred: 预测概率 (shape: (num_classes, m))

        Returns:
            loss: 平均交叉熵损失
        """
        m = Y_true.shape[1]

        # 添加小常数避免log(0)
        epsilon = 1e-15
        Y_pred_clipped = np.clip(Y_pred, epsilon, 1 - epsilon)

        # 交叉熵损失
        loss = -np.sum(Y_true * np.log(Y_pred_clipped)) / m
        return loss

    def backward_propagation(self, X, Y, cache):
        """
        反向传播算法

        Args:
            X: 输入数据 (shape: (n_features, m))
            Y: 真实标签 (shape: (num_classes, m))
            cache: 前向传播的缓存

        Returns:
            grads: 各层参数的梯度
        """
        grads = {}
        m = X.shape[1]
        L = self.num_layers - 1

        # 输出层梯度
        A_L = cache[f'A{L}']
        dZ_L = A_L - Y

        grads[f'dW{L}'] = np.dot(dZ_L, cache[f'A{L-1}'].T) / m
        grads[f'db{L}'] = np.sum(dZ_L, axis=1, keepdims=True) / m

        # 反向传播隐藏层
        for l in reversed(range(1, L)):
            dZ = np.dot(self.parameters[f'W{l+1}'].T, grads[f'dW{l+1}'] * m)
            dZ = dZ * self.relu_derivative(cache[f'Z{l}'])

            grads[f'dW{l}'] = np.dot(dZ, cache[f'A{l-1}'].T) / m
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m

        return grads

    def update_parameters(self, grads):
        """
        使用梯度下降更新参数

        Args:
            grads: 参数梯度字典
        """
        for l in range(1, self.num_layers):
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']

    def predict(self, X):
        """
        预测函数

        Args:
            X: 输入数据 (shape: (n_features, m))

        Returns:
            predictions: 预测类别 (shape: (m,))
            probabilities: 预测概率 (shape: (num_classes, m))
        """
        cache = self.forward_propagation(X)
        probabilities = cache[f'A{self.num_layers-1}']
        predictions = np.argmax(probabilities, axis=0)
        return predictions, probabilities

    def compute_accuracy(self, y_true, y_pred):
        """
        计算准确率

        Args:
            y_true: 真实标签 (shape: (m,))
            y_pred: 预测标签 (shape: (m,))

        Returns:
            accuracy: 准确率
        """
        return np.mean(y_true == y_pred)

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=128, verbose=10):
        """
        训练模型

        Args:
            X_train: 训练集特征 (shape: (n_features, m_train))
            y_train: 训练集标签 (shape: (m_train,))
            X_val: 验证集特征 (shape: (n_features, m_val))
            y_val: 验证集标签 (shape: (m_val,))
            epochs: 训练轮数
            batch_size: 批大小
            verbose: 每隔多少轮打印信息
        """
        print(f"🚀 开始训练MLP模型...")
        print(f"📊 训练集: {X_train.shape[1]} 样本")
        print(f"📊 验证集: {X_val.shape[1]} 样本")
        print(f"📈 网络结构: {self.layer_sizes}")
        print(f"🎯 学习率: {self.learning_rate}")
        print(f"⏱️  训练轮数: {epochs}")

        # 独热编码标签
        Y_train = self.one_hot_encode(y_train, self.layer_sizes[-1])
        Y_val = self.one_hot_encode(y_val, self.layer_sizes[-1])

        m_train = X_train.shape[1]
        num_batches = m_train // batch_size

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # 随机打乱训练数据
            permutation = np.random.permutation(m_train)
            X_train_shuffled = X_train[:, permutation]
            Y_train_shuffled = Y_train[:, permutation]

            epoch_loss = 0
            epoch_accuracy = 0

            # 小批量训练
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size

                X_batch = X_train_shuffled[:, start:end]
                Y_batch = Y_train_shuffled[:, start:end]

                # 前向传播
                cache = self.forward_propagation(X_batch)

                # 计算损失
                batch_loss = self.compute_loss(Y_batch, cache[f'A{self.num_layers-1}'])
                epoch_loss += batch_loss

                # 计算准确率
                y_pred_batch, _ = self.predict(X_batch)
                y_true_batch = np.argmax(Y_batch, axis=0)
                batch_accuracy = self.compute_accuracy(y_true_batch, y_pred_batch)
                epoch_accuracy += batch_accuracy

                # 反向传播
                grads = self.backward_propagation(X_batch, Y_batch, cache)

                # 更新参数
                self.update_parameters(grads)

            # 平均损失和准确率
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches

            # 验证集评估
            y_pred_val, _ = self.predict(X_val)
            val_accuracy = self.compute_accuracy(y_val, y_pred_val)
            val_loss = self.compute_loss(Y_val, self.forward_propagation(X_val)[f'A{self.num_layers-1}'])

            # 记录训练历史
            self.train_history['loss'].append(avg_loss)
            self.train_history['accuracy'].append(avg_accuracy)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_accuracy)

            epoch_time = time.time() - epoch_start_time

            # 打印训练信息
            if (epoch + 1) % verbose == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs:3d} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Acc: {avg_accuracy:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_accuracy:.4f} | "
                      f"Time: {epoch_time:.2f}s")

            # 早停机制 (如果验证准确率达到98%)
            if val_accuracy >= 0.98:
                print(f"\n🎉 提前停止! 验证准确率达到 {val_accuracy:.4f}")
                break

        total_time = time.time() - start_time
        print(f"\n✅ 训练完成!")
        print(f"⏱️  总训练时间: {total_time:.2f} 秒")
        print(f"📊 最终训练准确率: {avg_accuracy:.4f}")
        print(f"📊 最终验证准确率: {val_accuracy:.4f}")

        return self.train_history

    def plot_training_curves(self, save_path=None):
        """
        绘制训练曲线

        Args:
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        ax1.plot(self.train_history['loss'], label='Training Loss', color='blue', linewidth=2)
        ax1.plot(self.train_history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 准确率曲线
        ax2.plot(self.train_history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
        ax2.plot(self.train_history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 训练曲线已保存: {save_path}")

        plt.show()

    def evaluate_model(self, X_test, y_test):
        """
        评估模型性能

        Args:
            X_test: 测试集特征
            y_test: 测试集标签

        Returns:
            evaluation_results: 评估结果字典
        """
        print("\n🔍 模型评估...")

        y_pred, probabilities = self.predict(X_test)
        accuracy = self.compute_accuracy(y_test, y_pred)

        # 计算每类别的准确率
        class_accuracies = {}
        for digit in range(10):
            mask = y_test == digit
            if np.sum(mask) > 0:
                class_acc = self.compute_accuracy(y_test[mask], y_pred[mask])
                class_accuracies[digit] = class_acc

        print(f"📊 测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📊 每类别准确率:")
        for digit, acc in sorted(class_accuracies.items()):
            print(f"   数字 {digit}: {acc:.4f} ({acc*100:.2f}%)")

        evaluation_results = {
            'test_accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'total_parameters': self.count_parameters()
        }

        return evaluation_results

    def count_parameters(self):
        """计算模型参数数量"""
        total_params = 0
        for l in range(1, self.num_layers):
            total_params += self.parameters[f'W{l}'].size + self.parameters[f'b{l}'].size
        return total_params

def load_and_preprocess_mnist():
    """
    加载和预处理MNIST数据集

    Returns:
        X_train, y_train, X_test, y_test: 预处理后的数据
    """
    print("🔄 正在加载MNIST数据集...")

    try:
        # 尝试使用tensorflow加载
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        print("✅ 使用TensorFlow加载MNIST数据集")
    except ImportError:
        try:
            # 尝试使用scikit-learn的digits数据集作为替代
            from sklearn.datasets import fetch_openml
            print("🔄 使用OpenML加载MNIST数据集...")
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
            X, y = mnist.data, mnist.target.astype(int)

            # 划分训练集和测试集
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=10000, random_state=42
            )

            # 重新塑形为28x28图像
            X_train = X_train.reshape(-1, 28, 28)
            X_test = X_test.reshape(-1, 28, 28)

            print("✅ 使用OpenML加载MNIST数据集")
        except Exception as e:
            print(f"⚠️  无法加载MNIST数据: {e}")
            print("🔄 生成模拟MNIST数据用于演示...")
            # 生成模拟数据
            np.random.seed(42)
            X_train = np.random.randint(0, 256, (60000, 28, 28), dtype=np.uint8)
            y_train = np.random.randint(0, 10, 60000)
            X_test = np.random.randint(0, 256, (10000, 28, 28), dtype=np.uint8)
            y_test = np.random.randint(0, 10, 10000)
            print("✅ 生成模拟MNIST数据集")

    print(f"📊 训练集: {X_train.shape[0]} 张图片")
    print(f"📊 测试集: {X_test.shape[0]} 张图片")

    # 数据预处理
    print("\n🔄 正在进行数据预处理...")

    # 1. 归一化: [0,255] -> [0,1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # 2. 扁平化: (m, 28, 28) -> (784, m)
    X_train_flat = X_train.reshape(X_train.shape[0], -1).T
    X_test_flat = X_test.reshape(X_test.shape[0], -1).T

    print(f"📊 数据预处理完成")
    print(f"📊 输入维度: {X_train_flat.shape[0]} (扁平化后)")
    print(f"📊 像素值范围: [{X_train_flat.min():.3f}, {X_train_flat.max():.3f}]")

    return X_train_flat, y_train, X_test_flat, y_test

def visualize_mnist_samples(X, y, save_path=None):
    """
    可视化MNIST样本

    Args:
        X: 图像数据 (shape: (784, m))
        y: 标签 (shape: (m,))
        save_path: 保存路径
    """
    # 恢复图像形状
    X_images = X.T.reshape(-1, 28, 28)

    # 随机选择一些样本进行可视化
    indices = np.random.choice(len(X_images), 16, replace=False)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        ax.imshow(X_images[idx], cmap='gray')
        ax.set_title(f'Label: {y[idx]}')
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 MNIST样本可视化已保存: {save_path}")

    plt.show()

def main():
    """主函数"""
    print("🚀 Lab5 Part A: MLP从零实现")
    print("=" * 60)

    # 设置随机种子
    np.random.seed(42)

    # 1. 加载和预处理数据
    X_train, y_train, X_test, y_test = load_and_preprocess_mnist()

    # 可视化数据样本
    visualize_mnist_samples(X_train, y_train, 'lab5/outputs/mlp_results/mnist_samples.png')

    # 2. 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train.T, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    X_train_split = X_train_split.T
    X_val_split = X_val_split.T

    print(f"\n📊 数据划分:")
    print(f"📊 训练集: {X_train_split.shape[1]} 样本")
    print(f"📊 验证集: {X_val_split.shape[1]} 样本")
    print(f"📊 测试集: {X_test.shape[1]} 样本")

    # 3. 创建和训练MLP模型
    print(f"\n🧠 创建MLP模型...")

    # 网络结构: [输入层, 隐藏层, 输出层]
    # 784 -> 128 -> 10 (MNIST: 784维输入, 10类输出)
    layer_sizes = [784, 128, 10]

    mlp = MLPFromScratch(
        layer_sizes=layer_sizes,
        learning_rate=0.01,
        random_seed=42
    )

    # 训练模型
    training_history = mlp.train(
        X_train_split, y_train_split,
        X_val_split, y_val_split,
        epochs=100,
        batch_size=128,
        verbose=10
    )

    # 4. 绘制训练曲线
    mlp.plot_training_curves('lab5/outputs/mlp_results/training_curves.png')

    # 5. 测试集评估
    evaluation_results = mlp.evaluate_model(X_test, y_test)

    # 6. 保存结果
    print(f"\n💾 保存实验结果...")

    # 保存训练历史
    np.save('lab5/outputs/mlp_results/training_history.npy', training_history)
    np.save('lab5/outputs/mlp_results/evaluation_results.npy', evaluation_results)

    print(f"✅ MLP实验完成!")
    print(f"📊 最终测试准确率: {evaluation_results['test_accuracy']:.4f} ({evaluation_results['test_accuracy']*100:.2f}%)")
    print(f"🔢 模型参数数量: {evaluation_results['total_parameters']:,}")

    # 检查是否达到目标
    target_accuracy = 0.90
    if evaluation_results['test_accuracy'] >= target_accuracy:
        print(f"🎉 成功达到目标! 准确率 {evaluation_results['test_accuracy']*100:.2f}% > {target_accuracy*100:.1f}%")
    else:
        print(f"⚠️  未达到目标! 准确率 {evaluation_results['test_accuracy']*100:.2f}% < {target_accuracy*100:.1f}%")

if __name__ == "__main__":
    main()