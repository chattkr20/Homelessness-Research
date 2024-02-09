print('LIGHT GBM REGRESSOR + GRID SEARCH')

from numpy import arange
import pandas as pd
import numpy as np
import matplotlib as plot
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import sklearn.metrics
import lightgbm as lgb

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
# df = df.iloc[:, [4, 5, 6, 8, 10, 11]]

# normalize
scaler = preprocessing.Normalizer(norm="l2")
# df = pd.DataFrame(scaler.fit_transform(df))

X = df.iloc[:, 1:]
X = X.astype('float')
y = df.iloc[:, 0]
y = y.astype('float')
# X = pd.DataFrame(scaler.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Define the LightGBM Regressor
lgb_regressor = lgb.LGBMRegressor()

# Define the hyperparameters to search
param_grid = {
    'learning_rate': arange(0.05, 0.95, 0.05),
    'n_estimators': arange(150, 1000, 50),
    'num_leaves': arange(5, 50, 5),
#     'learning_rate': [0.05],
#     'n_estimators': [1250],
#     'num_leaves': [6],
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

# Saving Model to a File
import pickle
model_pkl_file = "testingWithoutBalanceOfState/noBalance_models/lgbm_model.pkl"
with open(model_pkl_file, 'wb') as file:  
    pickle.dump(best_lgb_model, file)

# Load Model from File, Get Predictions
import pickle
model_pkl_file = "testingWithoutBalanceOfState/noBalance_models/lgbm_model.pkl"
with open(model_pkl_file, 'rb') as file:
    best_lgb_model = pickle.load(file)
y_pred = best_lgb_model.predict(X_test)

# Evaluate the model
mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print("R2 Value on Test Set:", r2)


# Access feature importances
feature_importances = best_lgb_model.feature_importances_
print("Feature importances:", feature_importances)

# Images and Graphs
import matplotlib as plt
import matplotlib.pyplot

fig, axs = matplotlib.pyplot.subplots(1, figsize=(10, 7))  # Create a figure containing a single axes.

axs.plot(arange(0, y_pred.size), y_pred[np.argsort(y_test)], 'b.', arange(0, y_test.size), np.sort(y_test), 'r.')
# axs.plot(y_test, y_pred, 'b.')
# axs[1].plot(arange(0, y_pred.size), np.subtract(y_pred, y_test), 'b.')

matplotlib.pyplot.show()