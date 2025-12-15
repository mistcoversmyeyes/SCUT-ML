**Lec 4: 逻辑回归与 Softmax (Logistic Regression 考点：**
- **逻辑回归 (二分类)**：
    * [x] **Sigmoid 函数**：$g(z) = \frac{1}{1+e^{-z}}$，性质：$g'(z) = g(z)(1-g(z))$。
    * [x] **交叉熵损失 (Cross-Entropy/Log Loss)**：
        * 公式：$L = -\frac{1}{n}\sum [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]$。
        * 物理意义：源自最大似然估计 (MLE)。
- **Softmax 回归 (多分类)**：
    * [ ] **Softmax 函数**：$P(y=j|x) = \frac{e^{w_j^T x}}{\sum_{k} e^{w_k^T x}}$。
    * [ ] **多分类损失**：$L = -\sum_{j} \mathbb{I}(y=j) \log(p_j)$。

下面以二分类为例，介绍逻辑回归的基本内容。
## 数据
训练数据集 $D = \{(x_i, y_i)\}_{i=1}^N$，其中 $x_i \in \mathbb{R}^d$，$y_i \in \{0, 1\}$。


## 模型

### 定义
- 输入 $\vec{x_i}$ , 输出一个概率值 $\hat{y_i}$ 表示 $x_i$ 属于正类的概率。
- 函数模型: $f(x) = g(w^T x + b)$，其中 $g(z) = \frac{1}{1+e^{-z}}$ 是 Sigmoid 函数，其导数为 $g'(z) = g(z)(1-g(z))$。
- 

### 模型推导（大白话）
思考过程：
- 目标，预测一个 $(0,1)$ 之间的概率值。
- 考虑已有工具：线性回归模型 $f(x) = w^T x + b$，输出是实数，不符合概率定义。
- 解决方案：引入 Sigmoid 函数 $g(z) = \frac{1}{1+e^{-z}}$，将实数映射到 $(0,1)$。

> 数学推导过程（可跳过）：
> 先设我们想要预测的输出概率值为 $\hat{y_i}$ , 其值域为 $(0,1)$, 直觉上想要将其值域变换为 (-∞, +∞) , 我们最好将其拆分为两步:
> 1. 先将 $\hat{y_i}$ 映射到 (0, +∞) , 这一步可以使用 $\frac{\hat{y_i}}{1 - \hat{y_i}}$ 来实现.
> 2. 再将 (0, +∞) 映射到 (-∞, +∞) , 这一步可以使用 $\log$ 来实现.
> 综上, 我们可以得到如下的映射关系:
> $$
>\begin{aligned}
>     & \vec{w}^T \vec{x_i} + b = \log \frac{\hat{y_i}}{1 - \hat{y_i}} \\
> \Longleftrightarrow\ &z_i = \log \frac{\hat{y_i}}{1 - \hat{y_i}} \\
> \Longleftrightarrow\ & \hat{y_i} = \frac{1}{1 + e^{-z_i}} \\
> \Longleftrightarrow\ & \hat{y_i} = \frac{1}{1 + e^{-(\vec{w}^T \vec{x_i} + b)}} \\
> \end{aligned}
>$$
> 其中 $z_i$ 的值域为 $(-\infty, +\infty)$。

## 评估（损失函数——交叉熵损失）
### 定义
- 交叉熵损失函数 (Cross-Entropy Loss) $$ L = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \hat{y_i} + (1 - y_i) \log(1 - \hat{y_i}) \right] $$   

### 推导
- 交叉熵损失源自最大似然估计 (Maximum Likelihood Estimation, MLE)。
- 假设样本独立同分布 (i.i.d.)，则似然函数为 
  $$ 
  \begin{aligned}
      L(\vec{w}, b) &= \prod_{i=1}^N P(y_i | x_i; \vec{w}, b) \\
        &= \prod_{i=1}^N \hat{y_i}^{y_i} (1 - \hat{y_i})^{1 - y_i}
    \end{aligned}
  $$
- 取对数似然函数并取负号，得到交叉熵损失：
  $$
    \begin{aligned}
        -\log L(\vec{w}, b) &= -\sum_{i=1}^N \left[ y_i \log \hat{y_i} + (1 - y_i) \log(1 - \hat{y_i}) \right] \\
        &= N \cdot L
    \end{aligned}
  $$
- 其中，$\hat{y_i} = g(\vec{w}^T \vec{x_i} + b)$。



## 训练（梯度计算）

- 计算梯度：$$ \nabla_{\vec{w}} L = \frac{1}{N} \sum_{i=1}^N (\hat{y_i} - y_i) \vec{x_i} $$
- 计算偏导数：$$ \frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^N (\hat{y_i} - y_i) $$  
- 其中，$\hat{y_i} = g(\vec{w}^T \vec{x_i} + b)$。

## Q&A

### Q: 为什么使用对数几率比 (Log-Odds) 作为线性函数的输出？
A: 对数几率比将概率映射到实数范围，便于线性建模和优化。
[(Youtube)Why Do We Use Log-odds In Logistic Regression? - The Friendly Statistician
](https://www.youtube.com/watch?v=rDN3uvko2kw&embeds_referring_euri=https%3A%2F%2Fgemini.google.com%2F&embeds_referring_origin=https%3A%2F%2Fgemini.google.com&source_ve_path=MzY4NDIsMzY4NDIsMzY4NDIsMzY4NDIsMzY4NDIsMzY4NDIsMzY4NDIsMzY4NDIsMzY4NDIsMzY4NDIsMzY4NDIsMjg2NjY)

1. 为什么一定是 Logit？（深层理由）

   A. 统计学身份：指数族分布的“亲儿子”(Canonical Link)

   这是最根本的数学原因。逻辑回归假设样本服从伯努利分布（Bernoulli Distribution）。如果我们把伯努利分布的概率质量函数写成指数族分布（Exponential Family）的标准形式：

   $P(y|x) = p^y (1-p)^{1-y} = \exp\big( y \ln(\tfrac{p}{1-p}) + \ln(1-p) \big)$

   你会发现，自然参数（Natural Parameter）$\eta$ 恰好就是 $\ln\tfrac{p}{1-p}$。

   深意：使用对数几率作为链接函数（Link Function），使得模型在其自然参数上是线性的。这保证了充分统计量的存在，并且使得损失函数（负对数似然）是凸函数，保证了全局最优解。

   B. 几何对称性与“几率”的本质

   如果你只用“几率” $\text{Odds} = \tfrac{p}{1-p}$（范围 $[0, +\infty)$），它是不对称的：

   - $P=0.9 \rightarrow \text{Odds}=9$
   - $P=0.1 \rightarrow \text{Odds}=1/9 \approx 0.11$

   这就导致模型对“正类”和“负类”的敏感度不同。

   $\ln(x)$ 的作用：$\ln(9) \approx 2.2$，$\ln(1/9) \approx -2.2$。它将乘性的互逆关系（$9$ 和 $1/9$）转化为了加性的对称关系（$+2.2$ 和 $-2.2$）。这种对称性对于二分类问题至关重要。

2. 只有 Logit 吗？其他替代函数大比拼