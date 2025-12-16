Lec 12: 序列模型 (Sequence Modeling)
======

## 考点清单
- [ ] **RNN 问题**：梯度消失 (Vanishing Gradient) $\to$ 无法捕捉长距离依赖。
- [ ] **LSTM**：通过 **门 (Gates)** (遗忘门、输入门、输出门) 解决梯度消失。
- [ ] **Transformer (重点)**：
    - 核心：**Self-Attention** (自注意力机制)。
    - 公式：$Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$。
    - 相比 RNN 优势：可以并行计算 (Parallelizable)。


## RNN

### 输入编码 —— word embedding


### RNN 结构与计算



### RNN 训练


### RNN 的问题
1. 梯度消失 (Vanishing Gradient)：随着时间步的增加，梯度会指数级减小，导致模型难以学习长期依赖关系。
2. 梯度爆炸 (Exploding Gradient)：梯度值过大，导致参数更新不稳定。
3. 计算效率低：RNN 需要逐步处理序列，难以并行化。
4. 长期依赖捕捉困难：标准 RNN 难以捕捉长距离的依赖关系。



## Transformer

### POSITIONAL ENCODING
- 由于 Transformer 不具备序列顺序信息，需要通过位置编码 (Positional Encoding) 来引入位置信息。
- 常用的正弦和余弦函数编码方式：
  $$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}}) $$
  $$ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}}) $$  

### SELF-ATTENTION 机制


### 多头注意力 (Multi-Head Attention)


### 编码器-解码器结构 (Encoder-Decoder Architecture)

### 优势总结
1. 并行计算：Transformer 可以同时处理序列中的所有位置，显著提高计算效率。
2. 长距离依赖捕捉：Self-Attention 机制使得模型能够直接关注序列中任意位置的信息，改善了长期依赖问题。
3. 灵活性强：Transformer 结构可以方便地扩展和调整，适用于各种序列任务。
4. 优秀的性能：在机器翻译、文本生成等任务中，Transformer 已经成为主流模型，表现优异。



## 参考资料
- 《Attention Is All You Need》论文
- 课程录音与讲义
- 相关深度学习教材
- 在线资源与教程