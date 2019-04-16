# -*- coding: utf-8 -*-

#加载SVR模型算法库
from sklearn.svm import SVR
#训练集
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
#创建SVR回归模型的对象
clf = SVR()
# 利用训练集训练SVR回归模型
clf.fit(X, y)
"""
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
    gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
    tol=0.001, verbose=False)
"""
clf.predict([[1, 1]])