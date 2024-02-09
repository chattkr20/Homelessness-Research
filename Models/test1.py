print('ELASTIC NET REGRESSION + GRID SEARCH')

from numpy import arange
import pandas as pd
import numpy as np
import matplotlib as plot
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import sklearn.metrics
from scipy import stats

df = pd.read_csv("Models/data.csv")
array = df.to_numpy()

count = 0
for i in range(len(array)):
        for j in range(len(array[i])):
                if array[i][j] == "none":
                        array[i][j] = np.nan

df = DataFrame(array, columns=df.columns)
df = df.fillna(0.0) # !!!! Fills in NA values with 0, BIG assumption
df = df.iloc[:, 4:]

# normalize predictors
scaler = preprocessing.Normalizer()
df = pd.DataFrame(scaler.fit_transform(df))
df.to_csv("Models/normalized_data.csv")

X = df.iloc[:, 1:]
X = X.astype('float')
y = df.iloc[:, 0]
y = y.astype('float')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Define the Elastic Net Regressor
elastic_net = ElasticNet()

# Define the hyperparameters to search
param_grid = {
#     'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 0.5, 0.3, 0.4, 0.6, 0.7, 1.0, 10.0, 100.0],
#     'l1_ratio': arange(0, 1, 0.01),
#     'max_iter': [600, 800, 1000, 1200, 1400],
    'alpha': [1e-5],
    'l1_ratio': [0.71],
    'max_iter': [600],
}

# Create a GridSearchCV object
grid_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model
best_elastic_net = grid_search.best_estimator_

# Saving Model to a File
import pickle
model_pkl_file = "Models/Saved Models/elastic_net_model.pkl"
with open(model_pkl_file, 'wb') as file:  
    pickle.dump(best_elastic_net, file)

# Load Model from File, Get Predictions
import pickle
model_pkl_file = "Models/Saved Models/elastic_net_model.pkl"
with open(model_pkl_file, 'rb') as file:
    best_elastic_net = pickle.load(file)
y_pred = best_elastic_net.predict(X_test)

# Make predictions on the test set
y_pred = best_elastic_net.predict(X_test)

# Evaluate the model
mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print("R2 Value on Test Set:", r2)

print("Coefficients:", best_elastic_net.coef_)
print("Intercept:", best_elastic_net.intercept_)


# Images and Graphs
import matplotlib as plt
import matplotlib.pyplot
fig, axs = matplotlib.pyplot.subplots(1, figsize=(10, 7))  # Create a figure containing a single axes.
axs.plot(arange(0, y_pred.size), y_pred[np.argsort(y_test)], 'b.', arange(0, y_test.size), np.sort(y_test), 'r.')

# axs[1].plot(arange(0, y_pred.size), np.subtract(y_pred, y_test), 'b.')
matplotlib.pyplot.show()

# # Create a heatmap using seaborn
# import seaborn as sns
# import matplotlib.pyplot as plt
# df.columns = ["Homeless Count", "Avg. Rent Price","Poverty Count","Poverty Pct.","Unemployed Count","Unemployed Pct.","Education Count No HS Diploma","Education Count No College"]
# correlation_matrix = df.corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
# plt.title('Correlation Heatmap of Features')
# plt.show()

# # Feature Importance
# feature_importance = pd.Series(index = X_train.columns, data = np.abs(best_elastic_net.coef_))

# n_selected_features = (feature_importance>0).sum()
# print('{0:d} features, reduction of {1:2.2f}%'.format(
#     n_selected_features,(1-n_selected_features/len(feature_importance))*100))

# print(feature_importance)