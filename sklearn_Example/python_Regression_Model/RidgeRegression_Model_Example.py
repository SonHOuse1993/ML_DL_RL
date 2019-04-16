# -*- coding: utf-8 -*-

#加载线性模型算法库
from sklearn.linear_model import Ridge
# 创建岭回归模型的对象
reg = Ridge(alpha=.5)
# 利用训练集训练岭回归模型
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
#输出各个系数
reg.coef_
reg.intercept_

