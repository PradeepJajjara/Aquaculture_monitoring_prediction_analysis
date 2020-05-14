'''
Models implemented in Paper

BP Neural network - Implementation Done
LSTM - Implementation Done

call appropriate function for results
'''





import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import Sequential,layers
from nrmse import cal_nrmse
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

def read_dataset(file,type):
	if(type=='csv'):
		#print('file',file	)
		file = file + '.csv'
		df = pd.read_csv(file)
	elif(type=='excel'):
		df = pd.read_excel(file)
	return df


def normalize(dataframe):
	X = dataframe[['Temperature','PH','DO%']]
	x = X.values 
	min_max_scaler = MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	df = pd.DataFrame(x_scaled,columns=['Temperature','PH','DO%'])
	print(df.tail(5))
	return df

def neural_network():
	dataset = read_dataset('dataset','csv')
	df = normalize(dataset)
	y = np.array(df['DO%'])
	df.drop(columns='DO%',inplace=True)
	df.head()
	x = df.values
	folds = 2
	rmse_avg = []
	r2_avg = []
	nrmse_avg = []
	loss_avg = []
	folds = 1
	for i in range(folds):

		X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.2)
		print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
		model = Sequential()
		model.add(layers.Dense(256,activation = 'relu',input_shape = (X_train.shape[1],)))
		#model.add(layers.Dropout(0.3))
		model.add(layers.Dense(128,activation='relu'))
		model.add(layers.Dense(64,activation = 'relu'))
		#model.add(layers.Dropout(0.3))
		model.add(layers.Dense(32,activation = 'relu'))
		#model.add(layers.Dropout(0.3))
		model.add(layers.Dense(16, activation = 'relu'))
		model.add(layers.Dense(8, activation = 'relu'))


		model.add(layers.Dense(1,activation = 'linear'))

		sgd = keras.optimizers.SGD(lr =0.001,decay=1e-6,momentum = 0.9,nesterov = True)
		model.compile(optimizer = 'Adam',loss = 'mean_squared_error',metrics = ['mean_squared_error'])

		print(model.summary())

		count = 0
		history = model.fit(X_train,Y_train,epochs = 100,batch_size=64,validation_data = (X_test,Y_test))
		preds = model.predict(X_test)
		preds1 = model.predict(x)
		result = model.evaluate(X_test,Y_test) 
		loss = result[0]
		print(result)
		rmse_test = mean_squared_error(Y_test, preds)
		r2_test = r2_score(Y_test, preds)
		print("MSE of test set is {}".format(rmse_test))
		print("R score of test set is {}".format(r2_test))
		nrmse = cal_nrmse(Y_test,preds)
		print("nrmse of test set is {}".format(nrmse))
		loss_avg.append(loss)
		rmse_avg.append(rmse_test)
		r2_avg.append(r2_test)
		nrmse_avg.append(nrmse)
		with open('res_nn_paper.txt','a') as f:
			count = count + 1
			f.write("fold=%s\n" % str(count))
			f.write("rmse=%s\n" % str(rmse_test))
			f.write("loss=%s\n" % str(loss))
			f.write("r score=%s\n" % str(r2_test))
			f.write("nrmse=%s\n" % str(nrmse))
			f.write("-------------------------------------------------\n")
		#print("Mean_absolute_percentage_error of test set is",mean_absolute_percentage_error(Y_test,preds))

		plt.plot(history.history['mae'])
		plt.plot(history.history['val_mae'])
		plt.title('model mae')
		plt.ylabel('mae')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.show(block=False)
		plt.pause(1)
		plt.close()

		plt.plot(history.history['mse'])
		plt.plot(history.history['val_mse'])
		plt.title('model mse')
		plt.ylabel('mae')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.show(block=False)
		plt.pause(1)
		plt.close()

		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.show(block=False)
		plt.pause(1)
		plt.close()

		print('True 10 samples',Y_test[0:10])
		print('Predicted 10 samples',preds[0:10])
		arra = []
		arrak = []


		from tqdm import tqdm

		'''
		for i in tqdm(range(0,39988)):
		  arrak.append(i)  

		plt.gca().set_prop_cycle("color", ['blue','green'])
		plt.title('For DO% NN')
		plt.plot(np.array(arrak),y)
		plt.plot(np.array(arrak),np.array(preds1))
		plt.xlabel('Index')
		plt.ylabel('Values')
		plt.legend(['Normal data', ' Predicted Output'], loc='upper left')
		plt.show(block=False)
		plt.pause()
		plt.close()
		'''

	nrmse_avg = np.array(nrmse_avg)
	r2_avg = np.array(r2_avg)
	rmse_avg = np.array(rmse_avg)
	loss_avg = np.array(loss_avg)

	print("RMSE is {}".format(rmse_avg.mean()))
	print("R score is {}".format(r2_avg.mean()))
	print("loss_avgis{}".format(loss_avg.mean()))
	print("nrmse is {}".format(nrmse_avg.mean()))

def lstm():
	dataset = read_dataset('dataset','csv')
	df = normalize(dataset)
	y = np.array(df['DO%'])
	df.drop(columns='DO%',inplace=True)
	df.head()
	x = df.values
	count = 0
	rmse_avg = []
	r2_avg = []
	nrmse_avg = []
	loss_avg = []
	folds = 5
	my_learning_rate = 0.01
	for i in range(folds):
		model = Sequential()
		X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.2)
		print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

		es = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10, verbose=1, mode='auto',restore_best_weights = True)
		X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
		X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
		model.add(layers.LSTM(1024,input_shape = (X_train.shape[1],1)))
		#model.add(layers.Dense(2048,activation = 'relu'))
		#model.add(layers.Dropout(0.3))
		#model.add(layers.Dense(1024,activation = 'relu'))
		#model.add(layers.Dropout(0.3))
		model.add(layers.Dense(512, activation = 'relu'))
		model.add(layers.Dense(256,activation='relu'))
		model.add(layers.Dense(128,activation='relu'))
		#model.add(layers.Dropout(0.3))
		model.add(layers.Dense(64,activation = 'relu'))
		#model.add(layers.Dropout(0.3))
		model.add(layers.Dense(32,activation = 'relu'))
		#model.add(layers.Dropout(0.3))
		model.add(layers.Dense(16, activation = 'relu'))
		model.add(layers.Dense(8, activation = 'relu'))


		model.add(layers.Dense(1,activation = 'linear'))

		sgd = keras.optimizers.SGD(lr =0.001,decay=1e-6,momentum = 0.9,nesterov = True)
		model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),loss="mean_squared_error",metrics=[tf.keras.metrics.MeanSquaredError()])

		print(model.summary())


		history = model.fit(X_train,Y_train,epochs = 150,batch_size =64,validation_data = (X_test,Y_test),callbacks=[es])
		preds = model.predict(X_test)
		result = model.evaluate(X_test,Y_test) 
		loss = result[0]
		print(result)
		rmse_test = mean_squared_error(Y_test, preds)
		r2_test = r2_score(Y_test, preds)
		print("MSE of test set is {}".format(rmse_test))
		print("R score of test set is {}".format(r2_test))
		nrmse = cal_nrmse(Y_test,preds)
		print("nrmse of test set is {}".format(nrmse))
		loss_avg.append(loss)
		rmse_avg.append(rmse_test)
		r2_avg.append(r2_test)
		nrmse_avg.append(nrmse)
		with open('res_lstm_paper.txt','a') as f:
			count = count + 1
			f.write("fold=%s\n" % str(count))
			f.write("rmse=%s\n" % str(rmse_test))
			f.write("loss=%s\n" % str(loss))
			f.write("r score=%s\n" % str(r2_test))
			f.write("nrmse=%s\n" % str(nrmse))
			f.write("-------------------------------------------------\n")
		#print("Mean_absolute_percentage_error of test set is",mean_absolute_percentage_error(Y_test,preds))

		# plt.plot(history.history['mae'])
		# plt.plot(history.history['val_mae'])
		# plt.title('model mae')
		# plt.ylabel('mae')
		# plt.xlabel('epoch')
		# plt.legend(['train', 'val'], loc='upper left')
		# plt.show(block=False)
		# plt.pause(3)
		# plt.close()

		plt.plot(history.history["mean_squared_error"])
		plt.plot(history.history["val_mean_squared_error"])
		plt.title('model mse')
		plt.ylabel('mae')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.show(block=False)
		plt.pause(3)
		plt.close()

		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.show(block=False)
		plt.pause(3)
		plt.close()

	nrmse_avg = np.array(nrmse_avg)
	r2_avg = np.array(r2_avg)
	rmse_avg = np.array(rmse_avg)
	loss_avg = np.array(loss_avg)

	print("RMSE is {}".format(rmse_avg.mean()))
	print("R score is {}".format(r2_avg.mean()))
	print("loss_avgis{}".format(loss_avg.mean()))
	print("nrmse is {}".format(nrmse_avg.mean()))




#neural_network()
lstm()

