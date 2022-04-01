from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense
import joblib

diamonds = pd.read_csv('diamonds.csv')

diamonds = diamonds.loc[(diamonds['x'] > 0) | (diamonds['y'] > 0)]
diamonds.loc[11182, 'x'] = diamonds['x'].median()
diamonds.loc[11182, 'z'] = diamonds['z'].median()
diamonds = diamonds.loc[~((diamonds['y'] > 30) | (diamonds['z'] > 30))]
diamonds = pd.concat([diamonds, pd.get_dummies(diamonds['cut'], prefix='cut', drop_first=True)], axis=1)
diamonds = pd.concat([diamonds, pd.get_dummies(diamonds['color'], prefix='color', drop_first=True)], axis=1)
diamonds = pd.concat([diamonds, pd.get_dummies(diamonds['clarity'], prefix='clarity', drop_first=True)], axis=1)

from sklearn.decomposition import PCA
pca = PCA(n_components=1, random_state=123)
diamonds['dim_index'] = pca.fit_transform(diamonds[['x', 'y', 'z']])
diamonds.drop(['x', 'y', 'z'], axis=1, inplace=True)

X = diamonds.drop(['cut', 'color', 'clarity', 'price'], axis=1)
y = np.log(diamonds['price'])

numerical_features = ['carat', 'depth', 'table', 'dim_index']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X.loc[:, numerical_features] = scaler.fit_transform(X[numerical_features])

n_input = X.shape[1]
n_hidden1 = 32
n_hidden2 = 16
n_hidden3 = 8

nn_reg = Sequential()
nn_reg.add(Dense(units=n_hidden1, activation='relu', input_shape=(n_input,)))
nn_reg.add(Dense(units=n_hidden2, activation='relu'))
nn_reg.add(Dense(units=n_hidden3, activation='relu'))

nn_reg.add(Dense(units=1, activation=None))

batch_size = 32
n_epochs = 40
nn_reg.compile(loss='mean_absolute_error', optimizer='adam')
nn_reg.fit(X, y, epochs=n_epochs, batch_size=batch_size)

joblib.dump(pca, 'pca.joblib')
joblib.dump(scaler, 'scaler.joblib')
nn_reg.save('diamond-prices-model.h5')