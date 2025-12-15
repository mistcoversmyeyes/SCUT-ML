## 考点  PPT 2: 线性回归与梯度下降 (Linear Regression)
- [x] **模型与损失**：
    - 模型：$f(x) = w^T x + b$。
    - 损失：均方误差 $L(w) = \frac{1}{2}||y - Xw||^2$。
- [x] **解析解 (Closed-form)**：
    - 公式：$w^* = (X^T X)^{-1} X^T y$。
    - *条件*：$X^T X$ 必须可逆 (Invertible)。
- [x] **梯度下降 (Gradient Descent)**：
    - 迭代公式：$w_{k+1} = w_k - \eta \nabla L(w)$。
    - *手算准备*：给你简单数据，能算一步迭代。


## 模型
- 输入：训练数据集 $D = \{(x_i, y_i)\}_{i=1}^N$，其中 $x_i \in \mathbb{R}^d$，$y_i \in \mathbb{R}$。
- 记号约定（默认列向量）：$x_i, w \in \mathbb{R}^d$ 均为**列向量**，因此 $w^T x_i$ 为标量。
- 若使用设计矩阵 $X$：$X \in \mathbb{R}^{N\times d}$（第 $i$ 行为 $x_i^T$），$y \in \mathbb{R}^{N}$（列向量），故 $Xw$ 合法。
- 线性回归模型：$f(x) = w^T x + b $
- 目标：找到最优参数 $w$ 和 $b$，使得预测值尽可能接近真实值。


## 损失函数
均方误差 (Mean Squared Error, MSE) $$ L(w) = \frac{1}{2}||y - Xw||^2 = \frac{1}{2} \sum_{i=1}^N (y_i - f(x_i))^2 $$


## 解析解


## 梯度更新公式

- 梯度计算：$$ \nabla L(w) = -X^T(y - Xw) $$
- 参数更新公式 $$w_{k+1} = w_k - \eta \nabla L(w)$$
- 其中，$\eta$ 是学习率。   


## 梯度下降法推导
- 形式化问题:
  - 现有一函数 $Loss(\overrightarrow{w})$
  - 目标: 寻找 $\overrightarrow{w}$ 使得 $Loss(\overrightarrow{w})$ 最小化
- 记号说明：本节推导中 $a \cdot b$ 表示内积（列向量约定下等价于 $a^T b$）。
- 直觉:
  - 先随便猜测一个 $\overrightarrow{w_0}$, 然后计算该点的函数值 $Loss(\overrightarrow{w_0})$
  - 然后我们思考, 如何猜测下一个 $ \overrightarrow{w_1} $, 才能够使得 
  $$
  \begin{aligned}
            &L(\vec{w}_1) < L(\vec{w}_0) \\
  \Longleftrightarrow\ &L(\vec{w}_1)-L(\vec{w}_0) < 0 \\
  \Longleftrightarrow\ &\Delta L < 0 \\
  \Longleftrightarrow\ &\nabla L \cdot \Delta \vec{w} < 0 \\
  \Longleftrightarrow\ &\nabla L ||\Delta\vec{w} || \cdot \frac{\Delta \vec{w}}{||\Delta \vec{w}||} < 0 \\
  \Longleftrightarrow\ &\nabla L \cdot \frac{\Delta\vec{w}}{||\Delta \vec{w}||} < 0 \\
  \Longleftrightarrow\ & \nabla L (\vec{w_{1}} - \vec{w_0}) < 0 \\
  \end{aligned}
  $$
  - 从最后一步中的式子中, 我们知道了有关于 $\overrightarrow{w_1}$ 和 $\overrightarrow{w_0}$ 的关系:$$ \nabla L \cdot (\vec{w_1} - \vec{w_0}) < 0 $$
  - 这是一个约束, 只要我们的新的 $\overrightarrow{w_1}$ 满足这个约束, 那么就一定会使得 $Loss(\overrightarrow{w_1}) < Loss(\overrightarrow{w_0})$. 即新的 $\vec{w_1}$ **预测的结果更好**.
  - 然后有显然有 $$\nabla L \cdot -\lambda\nabla L < 0 $$
  - 也就是说,如果我们选择 $$ \begin{aligned} 
        &\Delta \vec{w} = -\lambda \nabla L & (\lambda>0) \end{aligned}$$ , 那么就一定会使得 $Loss(\overrightarrow{w_1}) < Loss(\overrightarrow{w_0})$.
  - 所以我们就得到了梯度下降的更新公式: 
    $$
    \begin{aligned} \\
        &\vec{w_1} - \vec{w_0} = -\lambda \nabla L & (\lambda>0) \\
    \Longleftrightarrow\ & \vec{w_1} = \vec{w_0} - \lambda \nabla L & (\lambda>0) \\
    \end{aligned} 
    $$




