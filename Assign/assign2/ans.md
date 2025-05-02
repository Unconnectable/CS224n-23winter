#书面

##a.

因为y是one-hot向量,只有一个位置是1，其他是0

也就是目标词$y_o=1$ ,其他的所有都是0,因此消去所有只剩下 $-\log(\hat{y}_o),$

##b.

###(i)过程参考第一课笔记

$$
注意到\hat{y}_w =P(w | c) \\
\frac{\partial J}{\partial v_c} = -u_o + \sum_w P(w \mid c) u_w\\
= \sum_w \hat{y}_wu_w-u_o\\

其中u_o=\sum_wy_wu_w,当y_w=y_o时为1 \\
原式=\sum_w (\hat{y}_w-y_w)u_w=U(\hat{y}_w-y_w)
$$



###(ii)

预测值和真实值相等时,也就是恰好预测了目标词=1,对其他词的预测值为0时

###(iii)

$\theta^{(t+1)} = \theta^{(t)} - \alpha \nabla_{\theta^{(t)}} J$

实际求出来的  $u_{预测值} -u_o$   是预测值和实际值的差的向量

$\theta_{old}减去了梯度$ 说明和输出向量更相似

### (iv)

如果有$\mathbf{u}_x = \alpha \mathbf{u}_y $   的情况,在归一化后$\mathbf{u}_x ,\mathbf{u}_y $  都会变成同一个向量,会丢失有用的信息,模型无法区分这两个词语了!



## c.

$J = \log \sum_w \exp (u_w^{T}v_c)-u_o$

$ w = o $  时:
$$
\log\sum_u\text{exp}(u_w^{T}v_c)=\log(\sum_{u \ne o}\text{exp}(u_w^{T}v_c)+\text{exp}(u_o^{T}v_c))\\
求导=\frac{\text{exp}(u_o^{T}v_c)·v_c}{\sum_u\text{exp}(u_w^{T}v_c)} \\
= \hat{y_o}v_c\\
答案=\hat{y_o}v_c-v_c=(\hat{y_o}-y_o)v_c,其中y_o=1\\
$$
当 $ w \neq o $ 时

同上
$$
\log\sum_u\text{exp}(k_w^{T}v_c)=\log(\sum_{u \ne w}\text{exp}(u_k^{T}v_c)+\text{exp}(u_w^{T}v_c))\\
答案=\frac{\text{exp}(u_w^{T}v_c)·v_c}{\sum_u\text{exp}(u_w^{T}v_c)} \\
=\hat{y}_w v_c=(\hat{y}_w-y_w) v_c ,其中y_w=0\\
$$


