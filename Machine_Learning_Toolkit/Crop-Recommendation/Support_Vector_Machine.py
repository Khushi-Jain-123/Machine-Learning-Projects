# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:32:47 2023

@author: khush
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

df = pd.read_csv("C:\\Users\\khush\\Downloads\\archive(1)\\Crop_recommendation.csv")

encoder = LabelEncoder()
encoder.fit(df["label"])
df["label"] = encoder.transform(df["label"])

features = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
target = df["label"]                               #98.8


#features = preprocessing.normalize(feature)        #90.22
#features = preprocessing.scale(feature)             #98.6
print(features)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=0)

SVM = SVC(kernel = "linear")
SVM.fit(Xtrain, Ytrain)
Ypred = SVM.predict(Xtest)

comp_df = pd.DataFrame({"actual": Ytest, "Predicted ": Ypred})
print(comp_df)

accuracy = metrics.accuracy_score(Ypred, Ytest)
print("Accuracy of SVM is:", accuracy)
