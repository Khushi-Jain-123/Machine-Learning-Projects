import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

df = pd.read_csv('C:\\Users\\khush\\Downloads\\boxoffice.csv', encoding = 'latin-1')

df.fillna('0', inplace = True)

df['domestic_revenue'] = df['domestic_revenue'].replace('[\$\,]', '', regex = True).astype(float)
df['opening_revenue'] = df['opening_revenue'].replace('[\$\,]', '', regex = True).astype(float)
df['budget'] = df['budget'].replace('[\$\,]', '', regex = True).astype(float)
df['world_revenue'] = df['world_revenue'].replace('[\$\,]', '', regex = True).astype(float)
df['opening_theaters'] = df['opening_theaters'].replace('[\,]', '', regex = True).astype(float)
df['release_days'] = df['release_days'].replace('[\,]', '', regex = True).astype(float)

for feature in df:
    if type(df[feature][0]) != str:
        print(feature)
        df[feature] = df[feature].replace(to_replace = 0, value = df[feature].mean(axis = 0))
        if feature == "world_revenue":
            continue
        if (feature=="budget" or feature == "opening_revenue"):
            df[feature] = np.log(df[feature])
        df[feature] = scale(df[feature])
df['MPAA'] = df['MPAA'].replace(to_replace = '0', value = 'Not Rated')
df['genres'] = df['genres'].replace(to_replace = '0', value = 'Action,Drama,Sci-Fi,Thriller')
df.head(25)


#plt.hist(df["MPAA"], color="blue")
plt.hist(df["opening_revenue"], color="blue")
"""plt.hist(df["domestic_revenue"], color="blue")
plt.hist(df["budget"], color="blue")
plt.hist(df["world_revenue"], color="blue")
plt.hist(df["opening_theaters"], color="blue")
plt.hist(df["release_days"], color="blue")
plt.hist(df["distributor"], color="blue")"""

mat = df.corr()

for i in range(len(df['genres'])): 
    df['genres'][i] = df['genres'][i].split(',')
df = pd.get_dummies( df.explode(column = ['genres']), columns=['genres']).groupby('title', as_index=False).sum()

x = df.drop(['world_revenue', 'title','release_days'], axis = 'columns')
y = df['world_revenue']

matx = x.corr()

from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(x_train, y_train)
y_pred = linear.predict(x_test)
y_pred

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print('error = ', mean_absolute_error(y_test, y_pred))
print('mean square error = ',metrics.mean_squared_error(y_test, y_pred))
print('accuracy = ', 1 - mean_absolute_error(y_test, y_pred))
print('accuracy of our model is: ', score*100)


