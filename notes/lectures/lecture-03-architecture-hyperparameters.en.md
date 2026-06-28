# Lecture 03: LM Architecture And Hyperparameters
Source: `external/cs336-lectures/lecture_03.pdf`

This is the English text companion. The primary learning artifact is the executable Chinese notebook:

- `notes/lectures/lecture-03-architecture-hyperparameters.zh.ipynb`

This Markdown file is for quick review and terminology lookup.

## Main Thread

这一讲从现代 Transformer 的常见组件出发，讨论大模型架构里哪些选择已经变成默认项，哪些仍然依赖实验经验。

## 1. 本讲路线图

Notebook pointer: section 1.

复习现代 Transformer，再看 RoPE、pre-norm、SwiGLU、bias、normalization、宽深比例和训练超参数。

## 2. 原始 Transformer 到现代变体

Notebook pointer: section 2.

现代 decoder-only LM 通常使用 causal self-attention、pre-norm、RoPE、SwiGLU 和无 bias 的线性层。

## 3. Normalization

Notebook pointer: section 3.

LayerNorm 和 RMSNorm 都用于稳定 activation 尺度；RMSNorm 更简单，少了均值中心化。

## 4. 位置编码

Notebook pointer: section 4.

RoPE 把位置信息注入 Q/K 的旋转结构中，让相对位置信息影响 attention 分数。

## 5. FFN 与 SwiGLU

Notebook pointer: section 5.

现代 LM 常用 gated FFN，例如 SwiGLU，用门控控制信息流。

## 6. 超参数不是装饰

Notebook pointer: section 6.

depth、width、heads、learning rate、batch size、初始化都会影响稳定性、效率和最终 loss。

## Checkpoints

- 能画出 decoder-only Transformer block 的主要路径。
- 能解释 pre-norm 为什么和训练稳定性有关。
- 能说出 RoPE 在什么位置作用。
- 能区分 architecture choice 和 hyperparameter choice。
