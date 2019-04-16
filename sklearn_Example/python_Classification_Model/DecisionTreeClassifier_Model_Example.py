# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
clf.fit(iris.data, iris.target)

print(clf.predict(iris.data))
print(clf.predict_proba(iris.data))

