from pandas import read_csv
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import numpy as np
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# load dataset
df = read_csv("dataset.csv")
Y = np.array(df['DO%'])
print(df.head())
df.drop(columns='DO%',inplace=True)

X = df.values

def my_model():
	model  = Sequential()
	# create model
	model.add(layers.Dense(256,activation = 'relu',input_shape = (X.shape[1],)))
	#model.add(layers.Dropout(0.3))
	model.add(layers.Dense(128,activation='relu'))
	model.add(layers.Dense(64,activation = 'relu'))
	#model.add(layers.Dropout(0.3))
	model.add(layers.Dense(32,activation = 'relu'))
	#model.add(layers.Dropout(0.3))
	model.add(layers.Dense(16, activation = 'relu'))
	model.add(layers.Dense(8, activation = 'relu'))
	model.add(layers.Dense(1,activation='linear'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model
seed = 7
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=my_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=2, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs=1)

print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))