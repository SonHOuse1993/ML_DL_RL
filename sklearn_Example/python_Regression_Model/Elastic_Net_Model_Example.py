# -*- coding: utf-8 -*-

#加载ElasticNet模型算法库
from sklearn.linear_model import ElasticNet
#加载数据集
from sklearn.datasets import make_regression
X, y = make_regression(n_features=2, random_state=0)
#创建ElasticNet回归模型的对象
regr = ElasticNet(random_state=0)
# 利用训练集训练ElasticNet回归模型
regr.fit(X, y)
print(regr.coef_)
print(regr.intercept_)
print(regr.predict([[0, 0]]))