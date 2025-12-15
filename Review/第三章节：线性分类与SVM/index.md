

## 考点 PPT 3: 线性分类与 SVM (Linear Classification & SVM)
- [ ] **SVM 核心**：最大化间隔 (Max Margin)。
- [ ] **Hinge Loss (必背公式)**：$\mathcal{L} = \sum \max(0, 1 - y_i(w^T x_i + b)) + \lambda ||w||^2$。
- [ ] **梯度下降变体 (必考)**：
    - **SGD**：每次用 1 个样本（快但震荡）。
    - **Mini-batch**：每次用一小批（平衡）。
    - **Batch**：每次用全部样本（稳但慢）。
- [ ] **[❌不考]**：SVM 对偶问题 (Dual Problem)、KKT 条件、核技巧 (Kernel Trick)。


## 模型
- 输入： 训练数据集 $D = \{(x_i, y_i)\}_{i=1}^N$，其中 $x_i \in \mathbb{R}^d$，$y_i \in \{-1, +1\}$。
- 记号约定（默认列向量）：$x_i, w \in \mathbb{R}^d$ 均为**列向量**，因此 $w^T x_i$ 为标量。
- 线性分类器：$f(x) = w^T x + b$
- 目标：找到最优参数 $w$ 和 $b$，使得 $$ y_i (w^T x_i + b) \geq 1, \quad \forall i $$

## 损失函数
- Hinge Loss：$$ \mathcal{L}(w, b) = \sum_{i=1}^N \max(0, 1 - y_i (w^T x_i + b)) + \frac{\lambda}{2} ||w||^2 $$
- 其中， $\lambda$ 是正则化参数，控制模型复杂度。
- 目标是最小化损失函数 $\mathcal{L}(w, b)$。
- 解释：
  - 第一项：惩罚分类错误或距离决策边界过近的样本。
  - 第二项：L2 正则化，防止过拟合。
- 优化方法：使用梯度下降及其变体（SGD、Mini-batch、Batch）来更新参数 $w$ 和 $b$。

## 梯度计算
- 对于单个样本 $(x_i, y_i)$，梯度为: 
  - 若 $y_i (w^T x_i + b) < 1$，则梯度为：$$ \nabla_w \mathcal{L} = -y_i x_i + \lambda w $$ $$ \nabla_b \mathcal{L} = -y_i $$
  - 若 $y_i (w^T x_i + b) \geq 1$，则梯度为零。
- 参数更新公式：
  - $w_{k+1} = w_k - \eta \nabla_w \mathcal{L}$
  - $b_{k+1} = b_k - \eta \nabla_b \mathcal{L}$
- 其中，$\eta$ 是学习率。


## 二维分类器示例

简单的直觉是，一个二位平面中的直线能够将整个平面分为两个部分，分别对应不同的类别。SVM 通过选择一条最佳的直线（决策边界），使得两类数据点之间的间隔最大化，从而提高分类的准确性和泛化能力。

为了避免“行向量/列向量”导致的转置歧义，这里统一采用**列向量**表示。

对于训练数据集 $D = \{(x_{i1}, y_i, class_i)\}_{i=1}^N$，其中 $x_{i1}, y_i \in \mathbb{R}$，$class_i \in \{-1, +1\}$。定义二维特征列向量

$$ x_i = [x_{i1}, y_i]^T \in \mathbb{R}^2, \quad w = [w_1, w_2]^T $$

则线性分类器为

$$ f(x_i) = w^T x_i + b $$

并希望对所有训练样本满足

$$ class_i \cdot f(x_i) = class_i\,(w^T x_i + b) \ge 1 $$

**大白话/直觉**：输出符号要和标签一致，并且离决策边界至少 1。
## 注意事项
- SVM 旨在最大化分类间隔，提升模型的泛化能力。
- 选择合适的正则化参数 $\lambda$ 和学习率 $\eta$
- 不同的梯度下降变体适用于不同的数据规模和计算资源。
- [ ] **[❌不考]**：SVM 对偶问题 (Dual Problem)、KKT 条件、核技巧 (Kernel Trick)。