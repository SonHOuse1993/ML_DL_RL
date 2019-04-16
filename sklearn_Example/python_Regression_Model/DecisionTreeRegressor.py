# -*- coding: utf-8 -*-

from sklearn.tree import  DecisionTreeRegressor
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = DecisionTreeRegressor()
clf = clf.fit(X, y)
clf.predict([[1, 1]])