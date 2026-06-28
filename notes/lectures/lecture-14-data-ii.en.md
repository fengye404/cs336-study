# Lecture 14: Data II
Source: `external/cs336-lectures/lecture_14.py`

This is the English text companion. The primary learning artifact is the executable Chinese notebook:

- `notes/lectures/lecture-14-data-ii.zh.ipynb`

This Markdown file is for quick review and terminology lookup.

## Main Thread

这一讲继续数据处理，重点是语言识别、数学数据构建、去重、Jaccard similarity、MinHash 和 LSH。

## 1. 本讲路线图

Notebook pointer: section 1.

从找特定语言/领域文本，到用相似度和哈希做大规模去重。

## 2. 语言与领域过滤

Notebook pointer: section 2.

训练数据通常要筛出特定语言、数学、代码或高质量网页。

## 3. 重复数据的危害

Notebook pointer: section 3.

重复会浪费 compute、改变分布，也可能加剧 benchmark contamination。

## 4. Jaccard similarity

Notebook pointer: section 4.

用集合交并比衡量文档相似度，常用于 shingles。

## 5. MinHash

Notebook pointer: section 5.

用多个随机哈希近似 Jaccard，避免两两比较所有文档。

## 6. LSH

Notebook pointer: section 6.

把相似文档放到同桶里，只比较候选近邻。

## Checkpoints

- 能手算 Jaccard similarity。
- 能解释 MinHash 为什么能近似 Jaccard。
- 能说出去重和 contamination 的关系。
