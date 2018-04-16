# -*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm

N = 100 # 每个类中的样本点
D = 2 # 维度
K = 3 # 类别个数
X = np.zeros((N*K,D)) # 样本input
y = np.zeros(N*K, dtype='uint8') # 类别标签
for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2  # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
# 可视化一下我们的样本点
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=mcm.Spectral)
plt.show()

