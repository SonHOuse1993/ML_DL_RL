# -*- coding: utf-8 -*-

import numpy as np
from sklearn.naive_bayes import MultinomialNB
X = np.random.randint(50, size=(1000, 100))
y = np.random.randint(6, size=(1000))

clf = MultinomialNB()
clf.fit(X, y)

print(clf.predict(X[2:3]))