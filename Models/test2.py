print('RANDOM FOREST REGRESSION + GRID SEARCH')

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
X = X.astype('float')
y = df.iloc[:, 0]
y = y.astype('float')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the RandomForestRegressor
rf_regressor = RandomForestRegressor()

# Define the hyperparameters and their possible values
param_grid = {
    'n_estimators': [10, 25, 50, 100, 150],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [1, 2, 5, 7, 10],
    'min_samples_leaf': [1, 2, 3, 4]
}

# Use GridSearchCV to find the best combination of hyperparameters
grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding mean squared error
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Negative Mean Squared Error: {:.2f}".format(grid_search.best_score_))

# Evaluate the model on the test set with the best hyperparameters
best_rf_regressor = grid_search.best_estimator_
test_mse = -grid_search.score(X_test, y_test)
print("Test Mean Squared Error with Best Hyperparameters: {:.2f}".format(test_mse))