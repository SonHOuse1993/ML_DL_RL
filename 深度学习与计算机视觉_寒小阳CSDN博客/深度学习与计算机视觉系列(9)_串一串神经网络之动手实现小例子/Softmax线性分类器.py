# -*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm

N = 100  # 每个类中的样本点
D = 2  # 维度
K = 3  # 类别个数
X = np.zeros((N*K, D))  # 样本input
y = np.zeros(N*K, dtype='uint8')  # 类别标签
for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2  # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
# 可视化一下我们的样本点
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=mcm.spectral)
plt.show()

# 训练线性分类器
# 随机初始化参数
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))

# 超参数的选择:需要自己敲定的步长和正则化系数
step_size = 1e-0
reg = 1e-3  # 正则化系数
# 梯度下降迭代循环
num_examples = X.shape[0]
for i in range(200):
    # 使用得分函数计算得分
    scores = np.dot(X, W) + b

    # 用指数函数还原
    exp_scores = np.exp(scores)
    # softmax归一化
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

    # 计算损失loss(包括互熵损失和正则化部分)
    # 计算log概率和互熵损失
    corect_logprobs = -np.log(probs[range(num_examples), y])  # 每个样本的data loss 就是选真实类别对应的那个p,计算-log(p)
    data_loss = np.sum(corect_logprobs)/num_examples
    #  加上正则化项
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss

    if i % 10 == 0:
        print("iteration %d: loss %f" % (i, loss))

    # 计算得分上的梯度
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples
    # 计算和回传梯度
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    # 正则化梯度
    dW += reg*W

    # 参数迭代更新
    W += -step_size * dW
    b += -step_size * db

# 评估准确度
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))

