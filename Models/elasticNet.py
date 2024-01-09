import pandas as pd
import numpy as np
import matplotlib as plot
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import sklearn.metrics as sm
import random
from sklearn import preprocessing

df = pd.read_csv("Models/data.csv")
array = df.to_numpy()

count = 0
for i in range(len(array)):
        for j in range(len(array[i])):
                if array[i][j] == "none":
                        array[i][j] = np.nan

df = DataFrame(array, columns=df.columns)
df = df.fillna(0.0) # !!!! Fills in NA values with 0, BIG assumption

scaled_df = preprocessing.normalize(df.iloc[:, 4:])
df = pd.DataFrame(scaled_df)
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

print(X)

# X = df.iloc[:, 5:]
# y = df.iloc[:, 4]

# sum = 0
# count = 1000
# vals, seeds = [], []
# for i in range(count):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= i)

#         enet = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=100)
#         enet.fit(X_train, y_train)

#         y_pred = enet.predict(X_test)
#         sum = (sum + sm.r2_score(y_test, y_pred)) / 2
#         vals.append(sm.r2_score(y_test, y_pred))
#         seeds.append(i)
# print(sum)
# print("Min at seed " + str(seeds[vals.index(min(vals))]) + ": " + str(min(vals)))
# print("Max at seed " + str(seeds[vals.index(max(vals))]) + ": " + str(max(vals)))