# -*- coding: utf-8 -*-

#加载Lasso模型算法库
from sklearn.linear_model import Lasso
# 创建Lasso回归模型的对象
reg = Lasso(alpha=0.1)
# 利用训练集训练Lasso回归模型
reg.fit([[0, 0], [1, 1]], [0, 1])
"""
Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
"""
# 使用测试集做预测
reg.predict([[1, 1]])