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

> **Insight：似然函数到底在“估计/衡量”什么？**
> 
> 把参数 $w$ 当作“可调旋钮”，把数据集 $D=\{(x_i,y_i)\}_{i=1}^N$ 当作“已经发生的事实”。
> 似然 $L(w)$ 衡量的是：**在当前参数 $w$ 下，模型让“我们观测到的整套标签 $y_1,\dots,y_N$”发生的概率有多大**。
> 换句话说，MLE 就是在所有 $w$ 里挑一个，让这套观测数据“最像是由模型生成的”。

对号入座（本特例为何写成连乘）：
- 逻辑回归把 $\hat{y_i}=\sigma(w^T x_i+b)$ 定义为 $P(y_i=1\mid x_i;w,b)$，因此 $P(y_i=0\mid x_i;w,b)=1-\hat{y_i}$。
- 若进一步假设样本在给定输入 $x_i$ 后条件独立（i.i.d. 的“独立”部分），则整套标签同时出现的概率可拆成各样本概率的乘积：$L(w)=\prod_{i=1}^N P(y_i\mid x_i;w,b)$。

**极大似然估计 (MLE) 推导：**

1.  对于随机变量 $Y$，其在 给定参数  $w$  和输入 $x_i$ 的条件下，其值为 $y_i$ 的概率为：
    $$
    P(Y=y_i\mid x_i)=p_i^{y_i}(1-p_i)^{1-y_i}=\begin{cases}
    p_i=\hat{y_i}, & y_i=1\\
    1-p_i=1-\hat{y_i}, & y_i=0
    \end{cases}.
    $$
    
    这个“统一式”还有一个重要作用：它易于取对数并改写为指数族形式，为后续推导 Logit（对数几率）与凸性的结论打下基础。

        
2.  似然函数（Likelihood）是所有样本概率的乘积 ：
    
    $$
    L\left(w\right)=\prod_{i=1}^{N} P\left(Y = y_{i}∣x_{i}\right)
    $$
    
3.  **负对数似然 (NLL) / 交叉熵损失 (Cross Entropy)**： 为了方便计算（变乘为加）并转化为最小化问题，我们取负对数 ：
    
    $$
    J\left(w\right)=−\frac{1}{N}\ln L\left(w\right)=−\frac{1}{N}\sum_{i=1}^{N} \left[y_{i}\ln \hat{y_i}+\left(1−y_{i}\right)\ln \left(1−\hat{y_i}\right)\right]
    $$
    
    这就是逻辑回归的损失函数。



## 训练（梯度计算）


我们需要计算损失  $J\left(w\right)$  对参数  $w$  的偏导数。

**前置性质（Sigmoid 的导数）：**

$$
g^{′}\left(z\right)=g\left(z\right)\left(1−g\left(z\right)\right)
$$

即： $\hat{y}^{′}=\hat{y}\left(1−\hat{y}\right)$ 。

**推导结论：** 虽然损失函数看起来很复杂（这就带有 log 又带有指数），但求导后的结果非常简洁优雅 ：

$$
\frac{\partial J\left(w\right)}{\partial w}=\frac{1}{N}\sum_{i=1}^{N} \left(\hat{y}_{i}−y_{i}\right)x_{i}
$$

*   **$\hat{y}_{i}−y_{i}$**：预测误差（Prediction Error）。
    
*   **$\vec{x}_{i}$**：输入特征。
    
*   **物理意义**：梯度方向就是 **(误差)  $\times$  (输入)**。误差越大，梯度更新幅度越大。
    

**参数更新 (Gradient Descent):**

$$
w:=w−\eta \frac{\partial J\left(w\right)}{\partial w}
$$

其中  $\eta$  是学习率 。

* * *

### 总结：逻辑回归 Cheat Sheet (考前速记)

| 概念            | 公式 / 核心点                       |
| --------------- | ----------------------------------- |
| 模型            | y^​=σ(wTx+b)=1+e−(wTx+b)1​          |
| 由来            | 对数几率ln(1−pp​)的线性假设         |
| 损失函数        | 交叉熵 (Cross Entropy):$$ J\left(w\right)=−\frac{1}{N}\ln L\left(w\right)=−\frac{1}{N}\sum_{i=1}^{N} \left[y_{i}\ln \hat{y_i}+\left(1−y_{i}\right)\ln \left(1−\hat{y_i}\right)\right]$$          |
| 优化方法        | 极大似然估计 (MLE)→梯度下降         |
| 梯度公式        | ​    $\frac{\partial J\left(w\right)}{\partial w}=\frac{1}{N}\sum_{i=1}^{N} \left(\hat{y}_{i}−y_{i}\right)x_{i}$        |
| 与 Softmax 关系 | Softmax 在类别数K=2时退化为逻辑回归 |


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