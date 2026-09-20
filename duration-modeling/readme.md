# 时长建模的工业界方案

## Weighted Logloss
论文：[Recommending What Video to Watch Next: A Multitask Ranking System](https://dl.acm.org/doi/10.1145/3298689.3346997)

将传统的二分类损失优化为加权损失，使得预测时长 $p$ 的最优解正好为 $\frac{t}{t + 1}$，从而反推出时长为 $\exp{(logit)}$。将时长 $t$ 转换为 $\frac{t}{t + 1}$，然后把它当作正例的label，负例的label则为 $\frac{1}{t + 1}$，加权损失函数为：

$$
L = - \frac{t}{t + 1} \log(p) - \frac{1}{t + 1} \log(1 - p)
$$

$$
\frac{\partial loss}{\partial p} = 0 \implies \text{最优解：} p = \frac{t}{t+1}
$$

其中 $p$ 是模型预测的时长，$t \in [0, +\infty)$ 是真实时长。在实际使用中，一般将分母中的 $t+1$ 去掉，不影响上述推导过程，即：

$$
L = - t \log(p) - \log(1 - p)
$$

在线上预估的时候，将预测logit转换为时长 $t = \exp{(logit)}$。