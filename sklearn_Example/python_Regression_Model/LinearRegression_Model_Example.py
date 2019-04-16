# -*- coding: utf-8 -*-

# 加载线性模型算法库
from sklearn import linear_model
# 创建线性回归模型的对象
regr = linear_model.LinearRegression()
# 利用训练集训练线性模型
regr.fit(X_train, y_train)
# 使用测试集做预测
y_pred = regr.predict(X_test)

