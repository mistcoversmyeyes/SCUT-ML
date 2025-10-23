#!/usr/bin/env python3
"""
Lab5 Part A: MLP从零实现 (简化测试版本)
快速验证算法逻辑，使用较小数据集
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体和绘图参数
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MLPFromScratchSimple:
    """简化版MLP实现，用于快速验证"""

    def __init__(self, layer_sizes, learning_rate=0.01, random_seed=42):
        np.random.seed(random_seed)
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        self.parameters = self.initialize_parameters()
        self.train_history = {'loss': [], 'accuracy': []}

    def initialize_parameters(self):
        """初始化参数"""
        parameters = {}
        for l in range(1, self.num_layers):
            parameters[f'W{l}'] = np.random.randn(
                self.layer_sizes[l], self.layer_sizes[l-1]
            ) * 0.01
            parameters[f'b{l}'] = np.zeros((self.layer_sizes[l], 1))
        return parameters

    def relu(self, Z):
        """ReLU激活函数"""
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        """ReLU导数"""
        return (Z > 0).astype(float)

    def softmax(self, Z):
        """Softmax激活函数"""
        Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def one_hot_encode(self, y, num_classes):
        """独热编码"""
        m = y.shape[0]
        Y_one_hot = np.zeros((num_classes, m))
        Y_one_hot[y, np.arange(m)] = 1
        return Y_one_hot

    def forward_propagation(self, X):
        """前向传播"""
        cache = {'A0': X}
        A = X

        for l in range(1, self.num_layers - 1):
            Z = np.dot(self.parameters[f'W{l}'], A) + self.parameters[f'b{l}']
            A = self.relu(Z)
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A

        Z_L = np.dot(self.parameters[f'W{self.num_layers-1}'], A) + \
              self.parameters[f'b{self.num_layers-1}']
        A_L = self.softmax(Z_L)
        cache[f'Z{self.num_layers-1}'] = Z_L
        cache[f'A{self.num_layers-1}'] = A_L

        return cache

    def compute_loss(self, Y_true, Y_pred):
        """计算交叉熵损失"""
        epsilon = 1e-15
        Y_pred_clipped = np.clip(Y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(Y_true * np.log(Y_pred_clipped)) / Y_true.shape[1]
        return loss

    def backward_propagation(self, X, Y, cache):
        """反向传播"""
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
            dZ = np.dot(self.parameters[f'W{l+1}'].T, dZ_L)
            dZ = dZ * self.relu_derivative(cache[f'Z{l}'])
            dZ_L = dZ

            grads[f'dW{l}'] = np.dot(dZ, cache[f'A{l-1}'].T) / m
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m

        return grads

    def update_parameters(self, grads):
        """更新参数"""
        for l in range(1, self.num_layers):
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']

    def predict(self, X):
        """预测"""
        cache = self.forward_propagation(X)
        probabilities = cache[f'A{self.num_layers-1}']
        predictions = np.argmax(probabilities, axis=0)
        return predictions

    def compute_accuracy(self, y_true, y_pred):
        """计算准确率"""
        return np.mean(y_true == y_pred)

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, verbose=5):
        """训练模型"""
        print(f"🚀 开始训练简化版MLP...")
        print(f"📊 训练集: {X_train.shape[1]} 样本")
        print(f"📊 验证集: {X_val.shape[1]} 样本")

        Y_train = self.one_hot_encode(y_train, self.layer_sizes[-1])
        Y_val = self.one_hot_encode(y_val, self.layer_sizes[-1])

        m_train = X_train.shape[1]

        for epoch in range(epochs):
            # 随机打乱数据
            permutation = np.random.permutation(m_train)
            X_train_shuffled = X_train[:, permutation]
            Y_train_shuffled = Y_train[:, permutation]

            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = 0

            # 小批量训练
            for i in range(0, m_train, batch_size):
                end = min(i + batch_size, m_train)
                X_batch = X_train_shuffled[:, i:end]
                Y_batch = Y_train_shuffled[:, i:end]

                # 前向传播
                cache = self.forward_propagation(X_batch)

                # 计算损失
                batch_loss = self.compute_loss(Y_batch, cache[f'A{self.num_layers-1}'])
                epoch_loss += batch_loss

                # 计算准确率
                y_pred_batch = self.predict(X_batch)
                y_true_batch = np.argmax(Y_batch, axis=0)
                batch_accuracy = self.compute_accuracy(y_true_batch, y_pred_batch)
                epoch_accuracy += batch_accuracy

                # 反向传播
                grads = self.backward_propagation(X_batch, Y_batch, cache)

                # 更新参数
                self.update_parameters(grads)

                num_batches += 1

            # 平均损失和准确率
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches

            # 验证集评估
            y_pred_val = self.predict(X_val)
            val_accuracy = self.compute_accuracy(y_val, y_pred_val)

            # 记录历史
            self.train_history['loss'].append(avg_loss)
            self.train_history['accuracy'].append(avg_accuracy)

            # 打印进度
            if (epoch + 1) % verbose == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs:3d} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Acc: {avg_accuracy:.4f} | "
                      f"Val Acc: {val_accuracy:.4f}")

        print(f"\n✅ 训练完成!")
        print(f"📊 最终验证准确率: {val_accuracy:.4f}")

        return self.train_history

def create_simple_mnist_data():
    """创建简化的MNIST模拟数据"""
    print("🔄 创建简化的MNIST模拟数据...")

    np.random.seed(42)

    # 创建一些基本的手写数字模式
    num_samples_per_digit = 100
    num_digits = 10
    image_size = 8  # 使用8x8小图像

    X = []
    y = []

    for digit in range(num_digits):
        for sample in range(num_samples_per_digit):
            # 创建基本模式
            pattern = np.zeros((image_size, image_size))

            # 根据数字添加简单模式
            if digit == 0:
                # 圆形
                for i in range(image_size):
                    for j in range(image_size):
                        if (i - 3.5)**2 + (j - 3.5)**2 < 9:
                            pattern[i, j] = 1
            elif digit == 1:
                # 垂直线
                pattern[:, 4] = 1
            elif digit == 2:
                # 添加简单模式
                pattern[2:6, 2:6] = 1
            elif digit == 3:
                # 添加简单模式
                pattern[1:7, 1:7] = 1
                pattern[2:6, 2:6] = 0
            elif digit == 4:
                # L形状
                pattern[2:6, 2] = 1
                pattern[6, 2:6] = 1
            else:
                # 随机模式
                pattern = np.random.rand(image_size, image_size) * 0.3

            # 添加噪声
            noise = np.random.rand(image_size, image_size) * 0.2
            pattern = pattern + noise

            # 添加一些变换
            if np.random.rand() > 0.5:
                pattern = np.fliplr(pattern)
            if np.random.rand() > 0.5:
                pattern = np.flipud(pattern)

            # 随机平移
            shift_x = np.random.randint(-1, 2)
            shift_y = np.random.randint(-1, 2)
            pattern = np.roll(np.roll(pattern, shift_x, axis=1), shift_y, axis=0)

            X.append(pattern.flatten())
            y.append(digit)

    X = np.array(X).T  # shape: (64, 1000)
    y = np.array(y)    # shape: (1000,)

    # 添加一些随机变化
    X = X + np.random.normal(0, 0.1, X.shape)
    X = np.clip(X, 0, 1)

    print(f"✅ 简化MNIST数据创建完成")
    print(f"📊 数据维度: {X.shape}")
    print(f"📊 标签维度: {y.shape}")
    print(f"📊 每个数字样本数: {num_samples_per_digit}")

    return X, y

def visualize_samples(X, y, save_path=None):
    """可视化样本"""
    X_images = X.T.reshape(-1, 8, 8)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for digit in range(10):
        mask = y == digit
        if np.sum(mask) > 0:
            idx = np.where(mask)[0][0]
            row, col = digit // 5, digit % 5
            axes[row, col].imshow(X_images[idx], cmap='gray')
            axes[row, col].set_title(f'Digit: {digit}')
            axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 样本可视化已保存: {save_path}")

    plt.show()

def main():
    """主函数"""
    print("🚀 Lab5 Part A: MLP从零实现 (简化测试版)")
    print("=" * 60)

    # 1. 创建简化数据
    X, y = create_simple_mnist_data()

    # 可视化样本
    visualize_samples(X, y, 'lab5/outputs/mlp_results/simple_samples.png')

    # 2. 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = X_train.T
    X_test = X_test.T

    # 3. 创建和训练MLP
    layer_sizes = [64, 32, 10]  # [输入层, 隐藏层, 输出层]

    mlp = MLPFromScratchSimple(
        layer_sizes=layer_sizes,
        learning_rate=0.1,
        random_seed=42
    )

    # 4. 训练模型
    history = mlp.train(
        X_train, y_train,
        X_test, y_test,
        epochs=30,
        batch_size=32,
        verbose=5
    )

    # 5. 最终评估
    y_pred = mlp.predict(X_test)
    final_accuracy = mlp.compute_accuracy(y_test, y_pred)

    print(f"\n📊 最终测试准确率: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

    # 6. 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lab5/outputs/mlp_results/simple_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 7. 保存结果
    np.save('lab5/outputs/mlp_results/simple_history.npy', history)

    print(f"✅ 简化版MLP实验完成!")
    print(f"🎯 测试准确率: {final_accuracy:.4f}")
    print(f"🔢 模型参数数量: {sum(mlp.parameters[f'W{l}'].size + mlp.parameters[f'b{l}'].size for l in range(1, mlp.num_layers))}")

if __name__ == "__main__":
    main()