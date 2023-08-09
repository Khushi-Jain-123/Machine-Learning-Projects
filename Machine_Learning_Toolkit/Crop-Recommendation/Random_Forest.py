# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 18:35:42 2023

@author: khush
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:\\Users\\khush\\Downloads\\archive(1)\\Crop_recommendation.csv")


features = df[["N","P","K","temperature", "humidity","ph", "rainfall"]]
target = df[["label"]]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=0)

randomforest = RandomForestClassifier(n_estimators=30, random_state=0)
randomforest.fit(Xtrain, Ytrain)
Ypred = randomforest.predict(Xtest)
print(Ypred)

#comp_df = pd.DataFrame({"Actual": Ytest, "predicted" : Ypred})

#print(comp_df)

accuracy = accuracy_score(Ypred, Ytest)
print("Accuracy of Random forest is:", accuracy)              #99.77