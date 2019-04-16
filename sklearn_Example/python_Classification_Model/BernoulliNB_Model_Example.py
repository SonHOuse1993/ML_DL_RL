# -*- coding: utf-8 -*-

import numpy as np
X = np.random.randint(50, size=(1000, 100))
y = np.random.randint(6, size=(1000))
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X, y)

print(clf.predict(X[2:3]))