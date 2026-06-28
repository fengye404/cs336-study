# 第 3 讲：语言模型架构与超参数
来源：`external/cs336-lectures/lecture_03.pdf`

这是 Lecture 03 的中文文本版。主学习材料是可执行 notebook：

- `notes/lectures/lecture-03-architecture-hyperparameters.zh.ipynb`

这个 Markdown 适合快速复习；notebook 按相同结构加入了可运行实验。

## 0. 本讲主线

这一讲从现代 Transformer 的常见组件出发，讨论大模型架构里哪些选择已经变成默认项，哪些仍然依赖实验经验。

## 1. 本讲路线图

对应 notebook：第 1 节。

复习现代 Transformer，再看 RoPE、pre-norm、SwiGLU、bias、normalization、宽深比例和训练超参数。

## 2. 原始 Transformer 到现代变体

对应 notebook：第 2 节。

现代 decoder-only LM 通常使用 causal self-attention、pre-norm、RoPE、SwiGLU 和无 bias 的线性层。

## 3. Normalization

对应 notebook：第 3 节。

LayerNorm 和 RMSNorm 都用于稳定 activation 尺度；RMSNorm 更简单，少了均值中心化。

## 4. 位置编码

对应 notebook：第 4 节。

RoPE 把位置信息注入 Q/K 的旋转结构中，让相对位置信息影响 attention 分数。

## 5. FFN 与 SwiGLU

对应 notebook：第 5 节。

现代 LM 常用 gated FFN，例如 SwiGLU，用门控控制信息流。

## 6. 超参数不是装饰

对应 notebook：第 6 节。

depth、width、heads、learning rate、batch size、初始化都会影响稳定性、效率和最终 loss。

## 检查点

- 能画出 decoder-only Transformer block 的主要路径。
- 能解释 pre-norm 为什么和训练稳定性有关。
- 能说出 RoPE 在什么位置作用。
- 能区分 architecture choice 和 hyperparameter choice。
