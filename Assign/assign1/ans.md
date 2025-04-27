# 作业 1：探索词向量

## Part 1: 基于计数的词向量

### 1.1

**对已有的 list 去重排序**

1. 对一个[str1,str2,str3...]分割
2. 列表推导:`listname = [x for x in range(100)]s`
3. 排序

```python
def distinct_words(corpus):
    corpus_words = []
    n_corpus_words = -1

    all_words = [word for document in corpus for word in document ]
    unique_words = set(all_words)
    corpus_words = sorted(unique_words)
    n_corpus_words = len(corpus_words)

    return corpus_words, n_corpus_words
```

### 1.2

举例
`["Start I Love Panda and move body END","Start I Love CS224n and Stanford END"]`

1. 首先去重然后映射为字典
2. 但是计算`co-occurence-matrix`必须在原来的`corpus`里面计算,比如这里`Start I`的结果应该是 2 而不应是 1，应为要分别计算
3. 对`corpus`的每个句子，直接计算左右范围然后遍历就行，注意`range`是左开右闭区间

### 注意:

1. M[i,j]的 i,j 是在新的 word2ind 的坐标
2. 无需重复计算 M[j,i]=M[i,j]

```python
def compute_co_occurrence_matrix(corpus, window_size=4):
    words, n_words = distinct_words(corpus)
    M = None
    word2ind = {}

    ### SOLUTION BEGIN
    leng = len(words)
    for i in range(leng):
        word2ind[words[i]] = i

    M = np.zeros((leng, leng))

    for sentence in corpus:
        for i, center_word in enumerate(sentence):
            center_idx = word2ind[center_word]
            left = max(0, i - window_size)
            right = min(len(sentence), i + window_size + 1)

            for j in range(left, right):
                if i == j:
                    continue
                context_idx = word2ind[sentence[j]]
                M[center_idx, context_idx] += 1
    ### SOLUTION END
```

### 1.3

#### `TruncatedSVD`初始化参数

- `n_components=k`：指定要保留的维度数（降维后的维度）
- `n_iter=n_iters`：指定迭代次数（影响算法的精确度）
- `random_state=0`：设置随机种子（保证结果可重现）

#### 2. 方法调用

- `fit_transform(M)`：一次性完成模型拟合和数据转换
  - `fit()`：学习数据的统计特性（计算奇异值分解）
  - `transform()`：应用降维转换

```py
def reduce_to_k_dim(M, k=2):
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    ### SOLUTION BEGIN

    svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=0)
    M_reduced = svd.fit_transform(M)
    ### SOLUTION END
```

### 1.4

```python
from gzip import FCOMMENT

def plot_embeddings(M_reduced, word2ind, words):
    ### SOLUTION BEGIN
    '''
        M_reduced : 二维词嵌入矩阵 (n_words, 2)
        word2ind : 单词到矩阵索引的映射字典
        words : 需要可视化的单词列表
    '''
    for word in words:
        idx = word2ind[word]
        #序列解包坐标
        x_coords, y_coords = M_reduced[idx]
          # 绘制红色"x"形散点(marker='x'指定形状)
        plt.scatter(x_coords, y_coords, marker="x", color="purple")
        # 在坐标点旁添加单词标签(fontsize控制字号)
        plt.annotate(word, (x_coords, y_coords), fontsize=12)

    plt.show()
    ### SOLUTION END
```

## 第二部分：基于预测的词向量

### 2.1/2.2

按照要求运行代码

### 2.3

```python
### SOLUTION BEGIN

w1 = "sofa"
w2 = "chair"
w3 = "beer"
w1_w2_dist = wv_from_bin.distance(w1, w2)
w1_w3_dist = wv_from_bin.distance(w1, w3)
#同义词 反义词
print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))

### SOLUTION END
```

### 2.4

```python
# Run this cell to answer the analogy -- man : grandfather :: woman : x

#pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'grandfather'], negative=['man']))
pprint.pprint(wv_from_bin.most_similar(positive=['water','wet'],negative=['sun']))
```

### 2.5

```python
### SOLUTION BEGIN

x, y, a, b = "usa", "washington", "china", "beijing"
assert wv_from_bin.most_similar(positive=[a, y], negative=[x])[0][0] == b

### SOLUTION END
```

### 2.6

```python
### SOLUTION BEGIN

x, y, a, b = "china","beijing","american",""
pprint.pprint(wv_from_bin.most_similar(positive=[a, y], negative=[x]))

### SOLUTION END
```

### 2.8

```python
### SOLUTION BEGIN

A = "man"
B = "woman"
word = "engineer"
pprint.pprint(wv_from_bin.most_similar(positive=[A, word], negative=[B]))
print()
pprint.pprint(wv_from_bin.most_similar(positive=[B, word], negative=[A]))

### SOLUTION END
```

### 2.9

#### a. 给出一个关于词向量中偏见是如何产生的解释。简要描述一个现实世界的例子来说明这种偏见的来源。

工程师,工人一般是男性,女性数量较少

#### b. 你可以使用什么方法来缓解词向量中表现出的偏见？简要描述一个现实世界的例子来说明这种方法。

补充数据，增加反刻板印象
