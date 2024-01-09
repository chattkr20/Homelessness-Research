print('LIGHT GBM REGRESSOR + GRID SEARCH')

import pandas as pd
import numpy as np
import matplotlib as plot
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

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

# Define the LightGBM Regressor
lgb_regressor = lgb.LGBMRegressor()

# Define the hyperparameters to search
param_grid = {
    'learning_rate': [0.01, 0.1, 0.5],
    'n_estimators': [50, 100, 200],
    'num_leaves': [20, 30, 40],
}

# Create a GridSearchCV object
grid_search = GridSearchCV(lgb_regressor, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model
best_lgb_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_lgb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)