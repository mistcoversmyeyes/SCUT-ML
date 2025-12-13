# 第四章 逻辑回归和多分类逻辑回归

## 序、机器学习的一般定式

首先，机器学习应当可以实现决策功能；其次，机器学习应该可以辅助开发者进行决策。

![](images/image-12.png)

![](images/image-13.png)

![](images/image-5.png)

## 一、Logistic Regression 逻辑回归

![](images/image-10.png)

![](images/image-11.png)

![](images/image-4.png)

![](images/image-3.png)

### （一） 二分问题

让我们举一个医学的例子，假设我们现在面前有一沓病历，每一本都有病人的健康信息，以及病人是否有心脏病的判定。我们的问题是：**如何推断一个人有心脏病的概率呢？**

![](images/image-9.png)

在第三章，我们学习了线性分类和线性回归，如图：

![](images/image-2.png)

接下来的内容如果直接看 PPT 会一脸懵逼，我来做一个省流讲解。

如果要做一个判定函数来判定一个人发生心脏病的概率，最简单的方式自然是 sign function（符号函数），如图所示：

![](images/Luban_17322046352404fa3db44-6acb-4e7d-bed4-8e4814051953.png)



但是，符号函数在梯度计算中存在一系列不方便的地方，所以科学家使用了 19 世纪期间解决人口模型的函数来代替符号函数，补充资料：[Logistic 函数浅谈](https://zhuanlan.zhihu.com/p/630739668)

在 [【五分钟机器学习】机器分类的基石：逻辑回归Logistic Regression\_哔哩哔哩\_bilibili](https://www.bilibili.com/video/BV1gA411i7SR?spm_id_from=333.788.recommend_more_video.-1\&vd_source=da1f2d78d73dfaae6ddfbcfe8e4801d8) 视频中提到，使用 sigmoid 函数来代替线性回归和符号函数，减少了极值对模型的影响，同时保持了模型分类的效果，WIN！

![](images/image-1.png)



![](images/image-6.png)

![](images/image.png)

![](images/image-8.png)

![](images/image-7.png)

按照惯例，我们就需要使用经典的损失函数去寻找一个好的函数，从而能让这一个函数能够尽量贴近 sign function 的效果。

**在这里，我们引入的新函数如图所示：**

![](images/image-21.png)

以下是**数学推导**：

![](images/image-22.png)

既然我们已经知道了逻辑损失的形式，那么我们只需要求解条件概率即可。

而任意的 (x\[i], y\[i]) 都是独立分布的，所以联合概率就相当于每一个条件概率相乘，如图所示：

![](images/image-20.png)

我们可以根据**最大似然估计（Maximum Likelihood Estimation）**&#x6765;估计出 w 的取值。

什么是最大自然估计？

最大似然估计（MLE）是一种统计方法，用于对给定数据集的潜在概率分布的参数进行推断。

我们可以利用最大似然估计简化连续乘法的表示形式，如图所示：

![](images/Luban_17322461142312c361abe-19da-4a06-ae96-d897aac6714b.png)

![](images/image-23.png)

和 SVM 类似，我们引入了正则化去简化模型并规避过拟合，如图所示：

![](images/image-19.png)

让我们对上一节的 SVM 和这一节的逻辑回归做一个对比：

我们可以发现，它们都是有监督的算法，都可以用来解决二分分类的问题

![](images/Luban_173224660777204dc147e-4f94-4212-9dc6-cde5619b3142.png)

![](images/image-14.png)

既然我们已经知道了损失函数，要求解 w 的最小值的话，我们只需要求导即可：

![](images/image-25.png)

![](images/Luban_1732246938162453cbf18-8e69-475c-bb9f-77a2812f4f95.png)

### （二） 多分类问题引论

此前的研究中，我们只考虑 y\[i] 属于 { -1, +1 } 的两种情况，但是万一 y\[i] 属于 {0, 1} 或者 y\[i] 属于 { 0, +1, ..., K - 1 } 的情况呢？

首先让我们先考虑最简单的情况，如图所示：

![](images/image-24.png)

我们类比此前的二分问题，也引用这一种估计方法来处理：

![](images/image-15.png)

![](images/image-16.png)

我们使用正则化去避免过拟合的情况：

![](images/image-17.png)

接下来就是大家最熟悉的求导，感兴趣的同学可以自行求解：

![](images/image-18.png)

## 二、Softmax Regression 多分类逻辑回归

![](images/image-26.png)

![](images/image-27.png)

![](images/image-28.png)

![](images/image-29.png)

![](images/image-30.png)

在之前的问题中，我们讨论的都是二分问题，那么如何拓展到多分类问题呢？在机器学习中，**映射到多个类的概率**可以实现“举一反三”的效果：

![](images/image-31.png)

对于每一个类 j 属于 {0, 1, ..., K - 1}，我们会定义一个权重向量 w\[j]，这个向量和类相对应，m 表示维度， K 表示类别：

![](images/image-39.png)

**这个红色的函数就被叫做 Softmax Function（多分类函数），红色函数的分母是一个归一化的辅助值。**

如果我们不采用这种形式，那么 max 不光滑，波动大！

根据矩阵乘法的相关原理，我们可以把公因数提取出来，如图所示：

![](images/image-37.png)

![](images/image-38.png)

![](images/image-32.png)

![](images/Luban_1732762306211715adc45-8b35-40b6-adc0-ce3bc172574d.png)

![](images/image-33.png)

## 三、多分类损失的变体（仅供了解）

## （一） Large-Margin Softmax Loss

![](images/image-34.png)

![](images/image-35.png)

![](images/image-36.png)

![](images/image-40.png)

## （二） Angular Softmax Loss（A-Softmax Loss）

![](images/image-41.png)

## （三） 为啥这些多分类损失都用了三角学和几何学的知识啊？推导和一般的 Softmax 有什么区别呢？

![](images/image-42.png)

# 第五章 欠拟合、过拟合、交叉验证

## 序、训练集、测试集和交叉验证集

![](images/image-43.png)

先回顾一下这三种损失函数，这三种损失函数可是期末的重点！

![](images/image-44.png)

![](images/image-45.png)

![](images/image-53.png)

这就涉及到一个灵魂拷问了：在已知数据集 D 的情况下，我们应该如何评价训练出来的模型效果好不好呢？

&#x20;     在拟合（ Regression ）问题上，我们可以用中学学过的平均值判定法或者最小方差损失判定法。

&#x20;     但是在很多问题上，不能像线性模型这么直观。这就引入接下来的三种集的概念了。

训练集：用于拟合模型的数据（标签已知） 70%&#x20;

测试集：用于评估模型的数据（Keep Secret） 30%

比较简单的处理数据集的方式就是将数据拆分成 训练集 和 测试集，比例大概为 7:3，我们只允许使用训练集进行模型的训练，利用训练集训练出的模型经过测试集的测试以后才能得到精准程度的评估。

以下是切分数据集合的代码示例

参考代码来源：https://machinelearningmastery.com/start-here/

这个博客十分推荐大家**一边复习一边去读一读！**

![](images/image-52.png)

后来，学术界认为只有训练集和测试集，就像红绿灯没有黄灯一样。所以学术界又从训练集中割了一块集合，名称交叉验证集，交叉验证集的作用可以让模型的训练有一个比对的样板，让模型变得更好，让我们用大家特别熟悉的高考举个例子：

![](images/diagram.png)

### &#x20;**必考：WHY WE NEED VALIDATION SET ?**

![](images/image-46.png)

![](images/image-47.png)

### 为什么机器学习需要交叉验证集？

#### 首先，是三个商业理由

① 我们需要选出最好的模型

② 我们需要评测被选择模型的准确性和强度

③ 我们最好需要评测 ROI of the modeling project（建模项目的投资回报率）

#### 其次，还有三个数据科学理由

① 我们学了这么多模型，不难发现，模型的构建主要是为了减少“损失”或者“偏差”

② 但是。在某种程度上，模型总是会拟合“噪声”以及 “信号”

③ 如果开发者仅仅局限在已知的数据集并且选择“最好”的那一个。那么很有可能就会导致过于乐观的情况出现

### 为什么高考训练需要大量的模拟考试？

高中三年的训练主要是为了减少在高考的失误率，但是，在某种程度上，我们不可否认的是，除了少数顶尖的名校能用出题人的思维训练极其有潜力的学生，其实很大多数同学复习的时候训练的点会“偏”。而如果我们仅仅局限在本校的考试中表现得好，万一又来一次 2022 年的高考数学全国一，怎么办呢？所以需要有交叉训练集（模考）来点醒同学的训练方向。

## 一、欠拟合和过拟合

我们老师上课的时候曾经说过：“**若无必要，无增其烦。**” 模型**最根本的作用就是获取数据的隐藏趋势**。

欠拟合（UnderFitting）：模型无法捕捉到数据的底层趋势

过拟合（OverFitting）：虽然模型完美拟合数据了，但是模型为了把噪音也纳入进来，导致表现形式很繁琐

![](images/image-54.png)

![](images/image-51.png)

![](images/image-48.png)

如何判定欠拟合和过拟合呢？一张表讲清楚：

|                                        | Underfitting（欠拟合）             | Overfitting（过拟合）                                    |
| -------------------------------------- | ----------------------------- | --------------------------------------------------- |
| Performance of training-set（训练集的表现）    | ERROR IS LARGE在训练集上表现不佳，错误多   | error is relatively small在训练集上表现极佳，错误极少             |
| Performance of generalization（通用问题的表现） | ERROR IS LARGE在新问题上表现不佳，错误多   | ERROR IS LARGE在新问题上表现极差，错误极多                        |
| Solution（遇到这种情况怎么办？）                   | ⬆️ Capacity (Complexity)复杂度增高 | ⬇️ Capacity (Complexity) 复杂度降低⬆️ Training Set 训练集增加 |
| Signs（特征）                              | HIGH BIAS（高偏差）                | HIGH VARIANCE（高方差）                                  |
| 大白话                                    | 训练集表现不佳和真实情况相差相差大             | 模型的错误振荡非常大有时候预测得好，有时候又预测不佳                          |

![](images/image-49.png)



What do we mean by the *variance&#x20;*&#x61;nd *bias&#x20;*&#x6F;f a statistical learning method?&#x20;

***Variance**&#x20;*&#x72;efers to the amount by which ***fˆ&#x20;*&#x20;would change** if we estimated it using a **diferent training data set**.

On the other hand, **bias** refers to the **error** that is introduced by **approximating a real-life problem**, which may be extremely complicated, by a much simpler model.



## 二、偏差-方差权衡（Bias-Variance Trade-off）

对于复杂的模型，过于复杂的模型可以忽略这个模型在未来数据的准确性，这种现象叫做偏差-方差权衡

低偏差：模型在训练集拟合得好

高方差：模型更有可能做出错误的决策

![](images/image-50.png)

小声 os：在特殊情况下，这种图会失效，但是考纲还是按照 PPT 为准。

想了解更多？请戳这里：https://mlu-explain.github.io/double-descent/

![我们的考纲只考亮的那一片，暗的那一片留给大家自行探索](images/image-67.png)

## 三、交叉验证

![](images/image-66.png)

如果我们想减少数据的可变性&#x20;

◼ 首先，我们可以使用不同分区进行多轮交叉验证

◼ 然后，对所有轮次的结果进行平均

给定条件：从总体 D 中采样的数据 𝑆



我们将 S 拆分成 K 个相等的 disjoint subsets 不相交子集 / 并查集（T\[1], ... , T\[K]）

随后，我们根据以下步骤求解出平均错误率：

![](images/image-63.png)

![](images/image-65.png)

![](images/image-64.png)

如果引入正则化的常数 λ 过大，所有的 θ 都会被惩罚并且变成 0，此时出现欠拟合的情况

如果引入正则化的常数 λ 适中，此时模型拟合效果较好

如果引入正则化的常数 λ 过小，此时出现过拟合的情况

![](images/image-55.png)

如何选取一个好的 λ 呢？

① 选取一个区间内可能的 λ 的取值，比如说

&#x20;for λ in range (0.02, 0.26, 0.02):

② 这就出现了 12个 不同的 λ 模型去校验

③ 对于每一个 λ\[i]，我们需要学习 θ\[i]，并且计算出 Jcv(θ\[i])

④ 选取 使得 Jcv(θ\[i]) 最小的 λ

⑤ 最后，我们汇报测试误差 Jtest(θ\[i])

&#x20;

**如何选取一个好的 λ 呢？（Using K-Fold Cross-Validation）**

① 将数据集拆分成训练集、交叉验证集和测试集

② 对于每一个有可能的值 λ，估计错误率

③ 选择 λ 可以让最小平均错误最小

④ 最终评估测试集的效果

# 第六章 非线性机器学习与集成方法

![](images/O~}S@~N$0\(1HX2]9PG0%XHR.png)

![](images/image-56.png)

![](images/image-57.png)

我们考试只考 ID3 模型（只适用于分类问题）

![](images/873O041VL0EGZMF$K5$[H_S.png)



## 一、决策树（重点考！！！）

让我们以打网球为例子，究竟什么时候我们才会出门打球呢？

![](images/image-58.png)

如果大家对**数据结构的“树”**&#x7ED3;构很清晰的话，我们将会发现判定出门打球的情况可以使用树进行表示，每一个**树的分支都代表一个可能的决策、结果或者反应。叶子节点表示最终结果。**

让我们用 Python 构造一个树吧：

![](images/image-59.png)

根据以上的代码，针对打网球的问题，我们总共有四种划分树的方式，如图所示：

![](images/image-60.png)

**究竟哪一种划分方式最好呢？**

划分方式**好**：对一个 value（节点），可以得到全为正的实例。其余 value（节点），可以得到全为负的实例。

划分方式**差**：没有区分度、属性对决策没有作用，每一个 value（节点）的正面实例和负面实例**数目都差不多（五五开）**

要是划分的**每一片叶子“区分度”**&#x90FD;足够大，就说明划分方式好。**Entropy（熵）**&#x5C31;是用于表示“区分度”的。

已知一个集合 D，这个集合 D 只有正向数据和负面数据。那么可以用以下公式计算 D 的熵：

![](images/image-61.png)

### 必考：如何求解数据集合 D 的熵？

解题口诀：**负数** **比例** **Alog2A**

假设这个叶子节点里面有 9 个正数节点，5 个 负数节点，根据口诀快速写出来：

![](images/image-62.png)

![](images/Luban_1732374875640bc614025-f5bd-42f7-aca3-13d94317257b.png)



我们自然可以从二种情况拓展到多种情况，如图所示

![](images/image-76.png)

在导论的时候，我们说，熵是体现信息密度的一个数学工具，也是体现集合纯度的工具。

我们再引入一个 Gain 函数用来表示获取信息的度量。

![](images/image-73.png)

Values(A)：对于属性 A 可以取的所有可能值的集合

Dv：是 D 的子集，表示属性 A 的值等于 v

![](images/image-74.png)

### 必考：把类似打网球的表格转化为决策树

有了公式，大家求解树的划分就简单很多了，详细的数学计算请自行训练。

![](images/image-75.png)

![](images/image-80.png)

回顾一下老师的拆分思路：

① **四棵树木套公式求解信息密度，发现 Outlook 的 Gain 值最大！！！**

② Outlook 中间节点没必要拆分，我们只需要拆分 Sunny 和 Rain 对应

③ 对于 Sunny 节点，我们可以用 Humidity, Wind, Temp 继续拆分，Humidity 的拆分效果最好，停止拆分

④ 对于 Rain 节点，我们可以用 Humidity, Wind, Temp 继续拆分，Wind 的拆分效果最好，停止拆分

![](images/image-77.png)

### 考试碎碎念

如果大题让我们目瞪法拆分，就不需要背公式，直接穷举强拆。

如果大题要求写出解题步骤，我们就需要严谨地按照公式走了。

## 二、集成学习

集成学习：将一系列基础模型合并到一起，从而产生一个更好的预测模型。

主要方法：Bagging（打包），Boosting（提升）

![](images/image-72.png)

随机森林是打包的一种拓展，但是这种方法使用了决策树作为基础的学习者

随机地从 p 个特征（features）中抽取出 m 个特征从而得到经过优化的划分特征

| Bagging | Random Forest |
| ------- | ------------- |
| 固定的结构   | 结构随机，训练效率更好   |

![](images/image-71.png)



![](images/image-79.png)

![](images/image-70.png)

![](images/image-81.png)

![](images/image-69.png)

![](images/image-78.png)

## 三、AdaBoost (Adaptive Boosting)

Combines base learners linearly（将基学习器线性组合）

Iteratively adapts to the errors made by base learners in previous iterations（迭代地适应前一轮中基学习器所犯的错误）

权重调动技巧：

**更高**的权重将会被分配到**未准确分类**的点、**更低**的权重将会被分配到**已准确分类**的点

![](images/image-68.png)

### Pytorch 实现简单的 AdaBoost

## 专栏：数学推导

[「五分钟机器学习」集成学习——Ensemble Learning - 哔哩哔哩](https://www.bilibili.com/opus/412213815564223304)

接下来，让**我们从样例入手，手把手实操 AdaBoost**

声明：手绘的 weight value 是随便写的，接下来的公式会告诉大家怎么精确地求解 weight value。&#x20;

![](images/Screenshot_20241127_135256_com.jideos.jnotes-1.png)

一直划分，直到达到低训练错误的阈值为止（理论上只要迭代的次数足够多，算法准确度可以到 100%）

换句话说，看看上面画的图，你是否发现蓝色的权重已经➗10了？

那么，只要我们把这三个线性的函数结合起来，是不是就可以拟合出非线性的情况了？

![](images/Screenshot_20241127_135256_com.jideos.jnotes.png)

以上只是三种模型的情况，如果扩展到 N 种模型会怎么样呢？

![](images/image-85.png)

接下来是公式总结，一定要记清楚哦：

![](images/image-83.png)

![](images/image-86.png)

请注意：更小错误率的分类器会变得更重要！

![](images/image-88.png)

![](images/image-82.png)

请注意，**符号函数是一个非线性函数，所以 AdaBoost 是可以处理非线性问题的。**

我们穷举了很多种分类方法，发现如果以 2.5 为分界线，小于 2.5 为 -1，大于 2.5 为 +1，错误率最小！

&#x20;        以下是最快解题的步骤，跳过了繁琐的求和！

![](images/Luban_17327571067730b885387-f69b-455f-a2ab-94798e05a154.jpeg)

为什么会使用 AdaBoost 呢？五大原因：

◼ 仅需要一个简单分类器作为基学习器 &#x20;

◼ 可以实现与强大分类器相似的预测 &#x20;

◼ 可以与任何学习算法结合 &#x20;

◼ 需要很少的参数调整 &#x20;

◼ 可以扩展到超出二分类的问题

小结一下，AdaBoost （Adaptive Boosting) 是集成方法中最流行且最有力的。AdaBoost 算法的重心放在了错误的数据点，实现简单。但是他依赖 base learner 的表现，并且在噪声数据面前显得十分脆弱。

## 四、GBDT ( Gradient Boosting Decision Tree，仅供了解 )

![](images/image-87.png)

![](images/image-84.png)

# 引入训练集、测试集、交叉验证集的绩点预测系统

Github 地址：https://github.com/kanghailong99/lab2

# 利用 Pytorch 解决 Softmax 多分类问题

# 面试鸭常考面试题

## 决策树面试题

https://www.mianshiya.com/bank/1821834636175642625/question/1821834647869362177

https://www.mianshiya.com/bank/1821834636175642625/question/1821834648129409026

https://www.mianshiya.com/bank/1821834636175642625/question/1821834648381067266

## 拓展：随机森林算法相关面试题 5032 — 5035

https://www.mianshiya.com/bank/1821834636175642625/question/1821834649727438849#heading-0

## 解释 Boosting 和 Bagging 算法的区别

https://www.mianshiya.com/bank/1821834636175642625/question/1821834650834735106

## AdaBoost 算法面试题

https://www.mianshiya.com/bank/1821834636175642625/question/1821834651107364865

https://www.mianshiya.com/bank/1821834636175642625/question/1821834651367411713

https://www.mianshiya.com/bank/1821834636175642625/question/1821834651702956033

https://www.mianshiya.com/bank/1821834636175642625/question/1821834651967197185

## 拓展：GBDT 相关面试题 5044 — 5046

https://www.mianshiya.com/bank/1821834636175642625/question/1821834653066104833

