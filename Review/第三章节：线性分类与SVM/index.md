**Lec3: 线性分类与 SVM (Linear Classification & SVM) 考点：**
- [ ] **SVM 核心**：最大化间隔 (Max Margin)。
- [ ] **Hinge Loss (必背公式)**：$\mathcal{L} = \sum \max(0, 1 - y_i(w^T x_i + b)) + \lambda ||w||^2$。
- [ ] **梯度下降变体 (必考)**：
    - **SGD**：每次用 1 个样本（快但震荡）。
    - **Mini-batch**：每次用一小批（平衡）。
    - **Batch**：每次用全部样本（稳但慢）。
- [ ] **[❌不考]**：SVM 对偶问题 (Dual Problem)、KKT 条件、核技巧 (Kernel Trick)。
----

## 模型
- 记号约定（默认列向量）：$x_i, w \in \mathbb{R}^d$ 均为**列向量**，因此 $w^T x_i$ 为标量。
- **输入:** 训练数据集 $D = \{(x_i, y_i)\}_{i=1}^N$，其中 $x_i \in \mathbb{R}^d$，$y_i \in \{-1, +1\}$。
- **线性分类器：** $f(x) = w^T x + b$
- **目标：** 找到最优参数 $w$ 和 $b$，使得 $$ y_i (w^T x_i + b) \geq 1, \quad \forall i $$

## 损失函数（评估）
- Hinge Loss：$$ \mathcal{L}(w, b) = \sum_{i=1}^N \max(0, 1 - y_i (w^T x_i + b)) + \frac{\lambda}{2} ||w||^2 $$
- 其中， $\lambda$ 是正则化参数，控制模型复杂度。
- 目标是最小化损失函数 $\mathcal{L}(w, b)$。
- 解释：
  - 第一项：惩罚分类错误或距离决策边界过近的样本。
  - 第二项：L2 正则化，防止过拟合。
- 优化方法：使用梯度下降及其变体（SGD、Mini-batch、Batch）来更新参数 $w$ 和 $b$。

## 梯度计算（训练）
### SGD 
- 对于单个样本 $(x_i, y_i)$，梯度为: 
  - 若 $y_i (w^T x_i + b) < 1$，则梯度为：$$ \nabla_w \mathcal{L} = -y_i x_i + \lambda w $$ $$ \nabla_b \mathcal{L} = -y_i $$
  - 若 $y_i (w^T x_i + b) \geq 1$，则梯度为零。
- 参数更新公式：
  - $w_{k+1} = w_k - \eta \nabla_w \mathcal{L}$
  - $b_{k+1} = b_k - \eta \nabla_b \mathcal{L}$
- 其中，$\eta$ 是学习率。

### Mini-batch 和 Batch 
- Mini-batch：对一小批样本计算平均梯度进行更新。
- Batch：对所有样本计算平均梯度进行更新。
- 更新公式类似于 SGD，只是梯度计算基于多个样本。

## 二维分类器示例

简单的直觉是，一个二位平面中的直线能够将整个平面分为两个部分，分别对应不同的类别。SVM 通过选择一条最佳的直线（决策边界），使得两类数据点之间的间隔最大化，从而提高分类的准确性和泛化能力。

对于训练数据集 $D = \{(x_i,y_i, class_{i})\}_{i=1}^N$，其中 $x_i \in R, \ y_i \in R, \ class_{i} \in \{-1, +1\}$，SVM 的目标是找到一个线性分类器 $f(x,y) = w_0x + w_1y + b = \overrightarrow{w} \cdot [x_i, y_i, 1]^T $，使得对于所有的训练样本，都满足以下条件： 
  - 当 $class_{i} = +1$ 时，$f(x_i,y_i) \geq 1$
  - 当 $class_{i} = -1$ 时，$f(x_i,y_i) \leq 1$
  - 即：$class_{i} \cdot f(x_i,y_i) \geq 1$
  - **大白话来说（或者直觉上来讲）**:我这个分类器 $$ f(x,y) = \overrightarrow{w}^T [x, y]^{T}  = w_0x + w_1y + b$$ 的输出值应当和样本的类别标签 $class_{i}$ 符号一致，并且距离决策边界至少为 1。
## 注意事项
- SVM 旨在最大化分类间隔，提升模型的泛化能力。
- 选择合适的正则化参数 $\lambda$ 和学习率 $\eta$
- 不同的梯度下降变体适用于不同的数据规模和计算资源。
- [ ] **[❌不考]**：SVM 对偶问题 (Dual Problem)、KKT 条件、核技巧 (Kernel Trick)。