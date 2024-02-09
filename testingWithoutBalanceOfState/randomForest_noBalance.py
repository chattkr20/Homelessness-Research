print('RANDOM FOREST REGRESSION + GRID SEARCH')

from numpy import arange
import pandas as pd
import numpy as np
import matplotlib as plot
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import sklearn.metrics as sm
import random
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import sklearn.metrics

df = pd.read_csv("testingWithoutBalanceOfState/Scratch - Sheet2.csv")
array = df.to_numpy()

count = 0
for i in range(len(array)):
        for j in range(len(array[i])):
                if array[i][j] == "none":
                        array[i][j] = np.nan

df = DataFrame(array, columns=df.columns)
df = df.fillna(0.0) # !!!! Fills in NA values with 0, BIG assumption
df = df.iloc[:, 4:]

X = df.iloc[:, 1:]
X = X.astype('float')
y = df.iloc[:, 0]
y = y.astype('float')


# normalize
scaler = preprocessing.Normalizer(norm="l2")
X = pd.DataFrame(scaler.fit_transform(X))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Define the RandomForestRegressor
rf_regressor = RandomForestRegressor()

# Define the hyperparameters and their possible values
param_grid = {
    'n_estimators': [30],
    'max_depth': [30],
    'min_samples_split': [3],
    'min_samples_leaf': [4],
    # 'n_estimators': [120],
    # 'max_depth': [4],
    # 'min_samples_split': [3],
    # 'min_samples_leaf': [3],
}

# Use GridSearchCV to find the best combination of hyperparameters
grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding mean squared error
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Negative Mean Squared Error: {:.2f}".format(grid_search.best_score_))

# Get predictions
best_rf_regressor = grid_search.best_estimator_
y_pred = best_rf_regressor.predict(X_test)

# Saving Model to a File
import pickle
model_pkl_file = "testingWithoutBalanceOfState/noBalance_models/random_forest_model.pkl"
with open(model_pkl_file, 'wb') as file:  
    pickle.dump(best_rf_regressor, file)

# Load Model from File, Get Predictions
import pickle
model_pkl_file = "testingWithoutBalanceOfState/noBalance_models/random_forest_model.pkl"
with open(model_pkl_file, 'rb') as file:
    best_random_forest = pickle.load(file)
y_pred = best_random_forest.predict(X_test)

# Evaluate the model
mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print("R2 Value on Test Set:", r2)


# Images and Graphs
import matplotlib as plt
import matplotlib.pyplot

fig, axs = matplotlib.pyplot.subplots(1, figsize=(10, 7))  # Create a figure containing a single axes.

axs.plot(arange(0, y_pred.size), y_pred[np.argsort(y_test)], 'b.', arange(0, y_test.size), np.sort(y_test), 'r.')
# axs.plot(y_test, y_pred, 'b.')
# axs[1].plot(arange(0, y_pred.size), np.subtract(y_pred, y_test), 'b.')

matplotlib.pyplot.show()

pd.DataFrame(y_pred).to_csv("testingWithoutBalanceOfState/results.csv")