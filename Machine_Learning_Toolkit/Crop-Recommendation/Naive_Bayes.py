# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:51:46 2023

@author: khush
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv("C:\\Users\\khush\\Downloads\\archive(1)\\Crop_recommendation.csv")

encoder = LabelEncoder()
encoder.fit(df["label"])
df["label"] = encoder.transform(df["label"])

feature = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
target = df["label"]                                    #99.09


#features = preprocessing.normalize(feature)            #96.13
features = preprocessing.scale(feature)                 #99.09
print(features)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target , test_size=0.2, random_state = 2)

naivebayes = GaussianNB()
naivebayes.fit(Xtrain, Ytrain)
Ypred = naivebayes.predict(Xtest)

comp_df = pd.DataFrame({"Actual": Ytest, "predicted" : Ypred})

print(comp_df)


print("mean absolute error:", metrics.mean_absolute_error(Ytest, Ypred))
print("mean squared error:", metrics.mean_squared_error(Ytest, Ypred))
print("Root mean squared error:", np.sqrt(metrics.mean_squared_error(Ytest, Ypred)))
accuracy = accuracy_score(Ytest, Ypred)
print("The accuracy of naive bayes classifier is :", accuracy)

print(classification_report(Ytest, Ypred))

plt.figure(figsize = (10,10))
plt.scatter(Ytest, Ypred, c="crimson")
p1 = max(max(Ypred), max(Ytest))
p2 = min(min(Ypred), min(Ytest))
plt.plot([p1,p2],[p1,p2], "b-")
plt.xlabel("True values", fontsize=15)
plt.ylabel("Predictions", fontsize=15)
plt.axis('equal')
plt.show()