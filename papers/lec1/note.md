# 1.Distributed Representations of Words and Phrases and their Compositionality



## 背景和动机

通过分布式向量表示(Word Embedding)捕捉词语间的**精确语法和语义关系**

### Skip-gram模型的问题

1.需要优化softmax

2.无法直接表示非组合词语

3.高频词影响训练速度和低频词表示质量



## 改进方法

1.训练时一定概率扔掉高频词语,提升训练速度和低频词的表示精度

2.**负采样(Negative Sampling)**代替softmax,

3.短语建模,解决skip-gram无法表示非组合性短语

通过统计共现频率识别高频词语,是用公式评分

**开源实现** :发布word2vec工具包

----



# 2.Efficient Estimation of Word Representations in Vector Space

看不懂...

