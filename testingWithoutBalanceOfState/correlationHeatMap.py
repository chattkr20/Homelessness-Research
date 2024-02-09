import pandas as pd
import numpy as np
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv("testingWithoutBalanceOfState/Scratch - Sheet2.csv")
array = df.to_numpy()

count = 0
for i in range(len(array)):
        for j in range(len(array[i])):
                if array[i][j] == "none":
                        array[i][j] = np.nan

df = DataFrame(array, columns=df.columns)
df = df.fillna(0.0) # !!!! Fills in NA values with 0, BIG assumption

df = df.iloc[:,4:]

scaler = preprocessing.Normalizer(norm="l2")
X = df.iloc[:, 1:]
X.columns = df.iloc[:,1:].columns
print(X)
X = pd.DataFrame(scaler.fit_transform(X))

correlation_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Features')
plt.show()