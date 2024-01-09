import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
import lightgbm

df = pd.read_csv("Models/data.csv")
array = df.to_numpy()
for i in range(len(array)):
        for j in range(len(array[i])):
                if array[i][j] == "none":
                        array[i][j] = np.nan

df = DataFrame(array, columns=df.columns)
df = df.fillna(0.0) # !!!! Fills in NA values with 0, BIG assumption

# scaled_df = preprocessing.normalize(df.iloc[:, 4:])
# df = pd.DataFrame(scaled_df)
# X = df.iloc[:, 1:]
# y = df.iloc[:, 0]

X = df.iloc[:, 5:]
y = df.iloc[:, 4]

vals, seeds = [], []
sum = 0
for i in range(200):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)

        model = lightgbm.LGBMRegressor(
                objective="regression",
                metric="l2",
                boosting_type="gbdt",
                n_estimators= 1400,
                num_leaves= 1000,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8)
        model.fit(X_train.astype('float').astype('int32'), y_train.astype('int32'))

        y_pred = model.predict(X_test.astype('float').astype('int32'))
        vals.append(sm.r2_score(y_test.astype('float').astype('int32'), y_pred.astype('float').astype('int32')))
        seeds.append(i)
        sum = sum + sm.r2_score(y_test.astype('float').astype('int32'), y_pred.astype('float').astype('int32'))

print("Min at seed " + str(seeds[vals.index(min(vals))]) + ": " + str(min(vals)))
print("Max at seed " + str(seeds[vals.index(max(vals))]) + ": " + str(max(vals)))
print("Average: " + str(sum / len(vals)))

fig, ax = matplotlib.pyplot.subplots()  # Create a figure containing a single axes.
ax.plot(seeds, vals)  # Plot some data on the axes.
matplotlib.pyplot.show()