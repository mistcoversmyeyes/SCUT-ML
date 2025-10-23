#!/usr/bin/env python3
"""
Lab5 Part B: CNN使用PyTorch实现
卷积神经网络 (Convolutional Neural Network) 完整实现

功能:
- 使用torchvision加载MNIST数据集
- 实现LeNet风格的CNN架构
- 完整的PyTorch训练与评估流程
- 与MLP的性能对比分析

目标: 测试集准确率 > 98%
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  使用设备: {device}")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置中文字体和绘图参数
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

class LeNetCNN(nn.Module):
    """
    LeNet风格的卷积神经网络
    经典的CNN架构，适合手写数字识别
    """

    def __init__(self, num_classes=10):
        super(LeNetCNN, self).__init__()

        # 第一个卷积块: 卷积层 + 激活 + 池化
        # 输入: 1x28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        # 输出: 6x24x24
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 输出: 6x12x12

        # 第二个卷积块
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # 输出: 16x8x8
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 输出: 16x4x4

        # 全连接层
        # 输入: 16x4x4 = 256
        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, 1, 28, 28)

        Returns:
            out: 输出张量 (batch_size, num_classes)
        """
        # 第一个卷积块
        x = self.conv1(x)      # -> (batch_size, 6, 24, 24)
        x = self.relu1(x)      # -> (batch_size, 6, 24, 24)
        x = self.pool1(x)      # -> (batch_size, 6, 12, 12)

        # 第二个卷积块
        x = self.conv2(x)      # -> (batch_size, 16, 8, 8)
        x = self.relu2(x)      # -> (batch_size, 16, 8, 8)
        x = self.pool2(x)      # -> (batch_size, 16, 4, 4)

        # 展平
        x = x.view(-1, 16*4*4)  # -> (batch_size, 256)

        # 全连接层
        x = self.fc1(x)        # -> (batch_size, 120)
        x = self.relu3(x)      # -> (batch_size, 120)
        x = self.fc2(x)        # -> (batch_size, 84)
        x = self.relu4(x)      # -> (batch_size, 84)
        x = self.fc3(x)        # -> (batch_size, 10)

        return x

    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class CNNTrainer:
    """
    CNN训练器类
    封装完整的训练和评估流程
    """

    def __init__(self, model, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'epoch_times': []
        }

    def train_epoch(self, train_loader):
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器

        Returns:
            avg_loss: 平均损失
            accuracy: 训练准确率
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, val_loader):
        """
        验证模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            avg_loss: 平均损失
            accuracy: 验证准确率
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = val_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(self, train_loader, val_loader, epochs=10, verbose=1):
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            verbose: 每隔多少轮打印信息
        """
        print(f"🚀 开始训练CNN模型...")
        print(f"📊 训练集批数: {len(train_loader)}")
        print(f"📊 验证集批数: {len(val_loader)}")
        print(f"🎯 学习率: {self.optimizer.param_groups[0]['lr']}")
        print(f"⏱️  训练轮数: {epochs}")
        print(f"🖥️  设备: {self.device}")
        print(f"🔢 模型参数数量: {self.model.count_parameters():,}")

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            epoch_time = time.time() - epoch_start_time

            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_accuracy'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_acc)
            self.train_history['epoch_times'].append(epoch_time)

            # 打印进度
            if (epoch + 1) % verbose == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs:3d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Time: {epoch_time:.2f}s")

            # 早停机制 (如果验证准确率达到99%)
            if val_acc >= 0.99:
                print(f"\n🎉 提前停止! 验证准确率达到 {val_acc:.4f}")
                break

        total_time = time.time() - start_time
        print(f"\n✅ 训练完成!")
        print(f"⏱️  总训练时间: {total_time:.2f} 秒")
        print(f"📊 最终验证准确率: {val_acc:.4f}")

        return self.train_history

    def evaluate(self, test_loader):
        """
        在测试集上评估模型

        Args:
            test_loader: 测试数据加载器

        Returns:
            results: 评估结果字典
        """
        print("\n🔍 模型评估...")

        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                # 每类别准确率
                c = (pred == target).squeeze()
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        test_loss /= len(test_loader)
        accuracy = correct / total

        print(f"📊 测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📊 每类别准确率:")
        class_accuracies = {}
        for i in range(10):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                class_accuracies[i] = acc
                print(f"   数字 {i}: {acc:.4f} ({acc*100:.2f}%)")

        results = {
            'test_accuracy': accuracy,
            'test_loss': test_loss,
            'class_accuracies': class_accuracies,
            'total_parameters': self.model.count_parameters()
        }

        return results

    def plot_training_curves(self, save_path=None):
        """
        绘制训练曲线

        Args:
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        ax1.plot(self.train_history['train_loss'], label='Training Loss', color='blue', linewidth=2)
        ax1.plot(self.train_history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 准确率曲线
        ax2.plot(self.train_history['train_accuracy'], label='Training Accuracy', color='blue', linewidth=2)
        ax2.plot(self.train_history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 CNN训练曲线已保存: {save_path}")

        plt.show()

def load_mnist_data(batch_size=64):
    """
    加载MNIST数据集

    Args:
        batch_size: 批大小

    Returns:
        train_loader, test_loader: 数据加载器
    """
    print("🔄 正在加载MNIST数据集...")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化 (MNIST均值和标准差)
    ])

    # 下载并加载训练集
    try:
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        print("✅ MNIST数据集下载成功")
    except Exception as e:
        print(f"⚠️  MNIST数据集下载失败: {e}")
        print("🔄 尝试使用本地数据...")
        # 如果下载失败，创建模拟数据
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)

        # 转换为torch tensor格式
        X = X.reshape(-1, 28, 28).astype(np.float32) / 255.0
        X = (X - 0.1307) / 0.3081
        y = y.astype(np.int64)

        # 创建TensorDataset
        train_size = 60000
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        train_dataset = TensorDataset(
            torch.from_numpy(X_train).unsqueeze(1),  # 添加通道维度
            torch.from_numpy(y_train)
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test).unsqueeze(1),
            torch.from_numpy(y_test)
        )
        print("✅ 使用OpenML数据创建数据集")

    # 划分验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"📊 训练集: {len(train_dataset)} 样本")
    print(f"📊 验证集: {len(val_dataset)} 样本")
    print(f"📊 测试集: {len(test_dataset)} 样本")
    print(f"📊 批大小: {batch_size}")

    return train_loader, val_loader, test_loader

def visualize_mnist_samples(dataloader, save_path=None):
    """
    可视化MNIST样本

    Args:
        dataloader: 数据加载器
        save_path: 保存路径
    """
    # 获取一个batch的数据
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < 10:
            ax.imshow(images[i][0], cmap='gray')
            ax.set_title(f'Label: {labels[i].item()}')
            ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 MNIST样本可视化已保存: {save_path}")

    plt.show()

def main():
    """主函数"""
    print("🚀 Lab5 Part B: CNN使用PyTorch实现")
    print("=" * 60)

    # 1. 加载数据
    train_loader, val_loader, test_loader = load_mnist_data(batch_size=64)

    # 可视化样本
    visualize_mnist_samples(train_loader, 'lab5/outputs/cnn_results/mnist_samples.png')

    # 2. 创建CNN模型
    print(f"\n🧠 创建CNN模型...")
    model = LeNetCNN(num_classes=10)
    print(f"🔢 模型参数数量: {model.count_parameters():,}")

    # 打印模型结构
    print("\n📋 模型结构:")
    print(model)

    # 3. 创建训练器
    trainer = CNNTrainer(model, device=device, learning_rate=0.001)

    # 4. 训练模型
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,
        verbose=2
    )

    # 5. 绘制训练曲线
    trainer.plot_training_curves('lab5/outputs/cnn_results/training_curves.png')

    # 6. 测试集评估
    evaluation_results = trainer.evaluate(test_loader)

    # 7. 保存结果
    print(f"\n💾 保存实验结果...")

    # 保存训练历史
    torch.save(training_history, 'lab5/outputs/cnn_results/training_history.pth')
    torch.save(evaluation_results, 'lab5/outputs/cnn_results/evaluation_results.pth')
    torch.save(model.state_dict(), 'lab5/outputs/cnn_results/cnn_model.pth')

    print(f"✅ CNN实验完成!")
    print(f"📊 最终测试准确率: {evaluation_results['test_accuracy']:.4f} ({evaluation_results['test_accuracy']*100:.2f}%)")
    print(f"🔢 模型参数数量: {evaluation_results['total_parameters']:,}")

    # 检查是否达到目标
    target_accuracy = 0.98
    if evaluation_results['test_accuracy'] >= target_accuracy:
        print(f"🎉 成功达到目标! 准确率 {evaluation_results['test_accuracy']*100:.2f}% > {target_accuracy*100:.1f}%")
    else:
        print(f"⚠️  未达到目标! 准确率 {evaluation_results['test_accuracy']*100:.2f}% < {target_accuracy*100:.1f}%")

    return evaluation_results

if __name__ == "__main__":
    results = main()