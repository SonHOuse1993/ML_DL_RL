# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(iris.data, iris.target)

print(neigh.predict(iris.data))

print(neigh.predict_proba(iris.data))