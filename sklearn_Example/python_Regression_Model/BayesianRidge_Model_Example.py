# -*- coding: utf-8 -*-

from sklearn.linear_model import BayesianRidge
X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
Y = [0., 1., 2., 3.]
reg = BayesianRidge()
reg.fit(X, Y)

