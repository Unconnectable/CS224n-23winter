# CS224n

## 使用的23winter版本 网站:[Stanford CS 224N](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/)

## 视频链接 [Stanford CS224N NLP](https://www.bilibili.com/video/BV1U5RNYgEfp/)

### **前置知识**

[Python Numpy 教程(含 Jupyter 和 Colab) )](https://cs231n.github.io/python-numpy-tutorial/)

## Schedule

Word Vectors

```
 for (int mask = 0; mask < (1 << (n - 1)); ++mask) {
    int sum = a[0];
    for (int i = 1; i < n; ++i) {
      if (mask & (1 << (i - 1))) {
        sum += a[i];
      } else {
        sum -= a[i];
      }
    }
    results.insert(sum);
  }
   vector<int> left(a.begin(), a.begin() + pos_eq);
    vector<int> right(a.begin() + pos_eq, a.end());

  for (int sum : left_sums) {
      if (right_sums.count(sum)) {
        return true;
      }
    }
```



