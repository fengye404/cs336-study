# 第 14 讲：Data II
来源：`external/cs336-lectures/lecture_14.py`

这是 Lecture 14 的中文文本版。主学习材料是可执行 notebook：

- `notes/lectures/lecture-14-data-ii.zh.ipynb`

这个 Markdown 适合快速复习；notebook 按相同结构加入了可运行实验。

## 0. 本讲主线

这一讲继续数据处理，重点是语言识别、数学数据构建、去重、Jaccard similarity、MinHash 和 LSH。

## 1. 本讲路线图

对应 notebook：第 1 节。

从找特定语言/领域文本，到用相似度和哈希做大规模去重。

## 2. 语言与领域过滤

对应 notebook：第 2 节。

训练数据通常要筛出特定语言、数学、代码或高质量网页。

## 3. 重复数据的危害

对应 notebook：第 3 节。

重复会浪费 compute、改变分布，也可能加剧 benchmark contamination。

## 4. Jaccard similarity

对应 notebook：第 4 节。

用集合交并比衡量文档相似度，常用于 shingles。

## 5. MinHash

对应 notebook：第 5 节。

用多个随机哈希近似 Jaccard，避免两两比较所有文档。

## 6. LSH

对应 notebook：第 6 节。

把相似文档放到同桶里，只比较候选近邻。

## 检查点

- 能手算 Jaccard similarity。
- 能解释 MinHash 为什么能近似 Jaccard。
- 能说出去重和 contamination 的关系。
