# -*- coding: utf-8 -*-

import numpy as np
from sklearn.svm import SVC
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

clf = SVC(C=1,kernel='rbf', gamma='auto')
clf.fit(X, y)

print(clf.predict([[-0.8, -1]]))

