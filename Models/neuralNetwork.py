import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


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


########################################################################

# build the model!

def buildModel(X_train):
        model = Sequential()
        model.add(Dense(1000, input_shape=(X_train.shape[1],), activation='relu')) # (features,)
        model.add(Dense(500, activation='relu'))
        model.add(Dense(250, activation='relu'))
        model.add(Dense(1, activation='linear')) # output node
        model.summary() # see what your model looks like

        # compile the model
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model

# early stopping callback
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=50,
                restore_best_weights = True)

########################################################################


vals, seeds = [], []
sum = 0
for i in range(1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        model = buildModel(X_train)

        history = model.fit(X_train.astype('float').astype('int32'), y_train.astype('int32'),
                    validation_data = (X_test, y_test),
                    callbacks=[es],
                    epochs=5000,
                    batch_size=50,
                    verbose=1)

        y_pred = model.predict(X_test.astype('float').astype('int32'))
        vals.append(sm.r2_score(y_test.astype('float').astype('int32'), y_pred.astype('float').astype('int32')))
        seeds.append(i)
        sum = sum + sm.r2_score(y_test.astype('float').astype('int32'), y_pred.astype('float').astype('int32'))

print("Min at seed " + str(seeds[vals.index(min(vals))]) + ": " + str(min(vals)))
print("Max at seed " + str(seeds[vals.index(max(vals))]) + ": " + str(max(vals)))
print("Average: " + str(sum / len(vals)))

# fig, ax = matplotlib.pyplot.subplots()  # Create a figure containing a single axes.
# ax.plot(seeds, vals)  # Plot some data on the axes.
# matplotlib.pyplot.show()