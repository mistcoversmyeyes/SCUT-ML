Lec 11: 神经网络与深度学习 (NN & DL)
====
- **基础组件**：
    * [ ] **激活函数**：ReLU (最常用，$max(0,x)$), Sigmoid, Tanh。
    * [ ] **反向传播 (BP)**：基于链式法则 (Chain Rule) $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$。
- **卷积神经网络 (CNN)**：
    * [ ] **卷积层 (Conv)**：局部连接，权值共享。需掌握 Kernel Size, Stride, Padding 对输出尺寸的影响。
    * [ ] **池化层 (Pooling)**：Max Pooling (取最大值), Average Pooling。作用是降维和提取主要特征。


## 数据


## 模型
### 单个神经元的组成
- 输入：特征向量 $x = [x_1, x_2, \ldots, x_n]$。
- 权重：$w = [w_1, w_2, \ldots, w_n]$。
- 偏置：$b$。
- 计算：线性组合 $z = w^T x + b$，然后将结果汇总通过激活函数 $a = f(z)$，得到神经元的输出。


### 神经网络的结构
#### 输入层 (Input Layer)：接收原始数据

输入：n 个 d 维特征向量，表示为一个 n×d 的矩阵。
权重矩阵：W，维度为 d×h，其中 h 是隐藏层神经元的数量。
偏置向量：b，维度为 1×h。
计算过程：
1. 线性变换：计算线性组合 Z = XW + b，其中 X 是输入矩阵。
2. 激活函数：对 Z 应用激活函数（如 ReLU 或 Sigmoid）得到隐藏层输出 A = f(Z)。   



#### 隐藏层 (Hidden Layers)：多个神经元组成，进行特征提取和转换。
#### 输出层 (Output Layer)：生成最终预测结果。

!!!note "补充说明"
    - 神经网络的层数和每层的神经元数量是超参数，需要通过实验调整。
    - 深度学习中的“深度”通常指隐藏层的数量较多。
    - 默认每一层神经网络之间都是全连接的。




## 训练与优化

## 