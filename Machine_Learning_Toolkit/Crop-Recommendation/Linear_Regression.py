import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


df = pd.read_csv("C:\\Users\\khush\\Downloads\\archive(1)\\Crop_recommendation.csv")
#print(df)

df1 = df.iloc[0:199,:]

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df["label"])
df["label"] = encoder.transform(df[["label"]])

features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size= 0.2, random_state= 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Xtrain, Ytrain)
print(regressor.coef_)


Ypred = regressor.predict(Xtest)
comp_df = pd.DataFrame({"Actual": Ytest, "Predicted": Ypred})
print(comp_df)
print("Mean Sqared Error:", metrics.mean_squared_error(Ytest,Ypred))
print("Root Mean Sqared Error:",np.sqrt(metrics.mean_squared_error(Ytest,Ypred)))


print(Ypred)
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