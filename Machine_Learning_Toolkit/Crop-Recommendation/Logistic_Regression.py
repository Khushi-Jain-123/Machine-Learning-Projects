# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:15:48 2023

@author: khush
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

df = pd.read_csv("C:\\Users\\khush\\Downloads\\archive(1)\\Crop_recommendation.csv")


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df["label"])
df["label"] = encoder.transform(df[["label"]])

features = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
target = df["label"]
                                                              #96.81

#features = preprocessing.normalize(feature)                  #84.77
#features = preprocessing.scale(feature)                      #96.36
print(features)

from sklearn.model_selection import train_test_split
Xtrain , Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=0)

regressor = LogisticRegression(random_state=2)
regressor.fit(Xtrain , Ytrain)
Ypred = regressor.predict(Xtest)


comp_df = pd.DataFrame({"Actual": Ytest, "predicted" : Ypred})

print(comp_df)

print("mean absolute error:", metrics.mean_absolute_error(Ytest, Ypred))
print("mean squared error:", metrics.mean_squared_error(Ytest, Ypred))
print("Root mean squared error:", np.sqrt(metrics.mean_squared_error(Ytest, Ypred)))

accuracy = metrics.accuracy_score(Ytest, Ypred)
print("logistic Regression's accuracy is:", accuracy)
print(classification_report(Ytest,Ypred))

plt.figure(figsize = (10,10))
plt.scatter(Ytest, Ypred, c="crimson")
plt.yscale('log')
plt.xscale('log')
p1 = max(max(Ypred), max(Ytest))
p2 = min(min(Ypred), min(Ytest))
plt.plot([p1,p2],[p1,p2], "b-")
plt.xlabel("True values", fontsize=15)
plt.ylabel("Predictions", fontsize=15)
plt.axis('equal')
plt.show()