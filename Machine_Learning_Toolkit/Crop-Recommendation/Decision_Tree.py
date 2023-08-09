# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:10:36 2023

@author: khush
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree
from sklearn import preprocessing

df = pd.read_csv("C:\\Users\\khush\\Downloads\\archive(1)\\Crop_recommendation.csv")

feature = df[["N","P","K","temperature", "humidity","ph", "rainfall"]]
target = df[["label"]]                                #98.4

#features = preprocessing.normalize(feature)           #90.9
features = preprocessing.scale(feature)              #98.4

print(features)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=0)

tree = DecisionTreeClassifier(criterion="entropy", max_depth=7, random_state=0)
tree.fit(Xtrain, Ytrain)
Ypred = tree.predict(Xtest)



accuracy = metrics.accuracy_score(Ypred, Ytest)
print("Decision Tree accuracy is:", accuracy)
plot_tree(tree, filled = 'true')

