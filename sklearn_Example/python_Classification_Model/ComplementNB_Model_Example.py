# -*- coding: utf-8 -*-

import numpy as np
from sklearn import naive_bayes

X = np.random.randint(50, size=(1000, 100))
y = np.random.randint(6, size=(1000))

clf = naive_bayes.ComplementNB()
clf.fit(X, y)

print(clf.predict(X[2:3]))