import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from nrmse import cal_nrmse
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
df= pd.read_csv('dataset.csv')
ddd = np.array(list(df['DO%']))
df.dropna(inplace=True)
arr = list(df.columns)
print(df.head(10))
print(df.isnull().sum())
print(df.columns)
print(df.info())
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score
x = df.values 
#y1 = df['DO%'].values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled,columns=arr)
print(df.tail(5))
print(df.columns)
dd = df
y = np.array(df['DO%'])
den = ddd.max() - ddd.min()
add = ddd.min()

#new_scaler = MinMaxScaler()
#scaled = new_scaler.fit_transform(y1)
df.drop(columns='DO%',inplace=True)
df.head()
x = df.values
from sklearn.utils.validation import check_array


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

'''
import tensorflow as tf
from tensorflow import keras
from keras import Sequential,layers
model = Sequential()
# X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
# model.add(layers.LSTM(128,input_shape = (X_train.shape[1],1)))
#model.add(layers.Dropout(0.3))
model.add(layers.Dense(256,activation = 'relu',input_shape = (X_train.shape[1],)))
#model.add(layers.Dropout(0.3))
model.add(layers.Dense(128,activation = 'relu'))
#model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(32, activation = 'relu'))

model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(8, activation = 'relu'))

#model.add(layers.Dropout(0.3))
model.add(layers.Dense(1,activation = 'linear'))

sgd = keras.optimizers.SGD(lr =0.001,decay=1e-6,momentum = 0.9,nesterov = True)
model.compile(optimizer = 'adam',loss = 'mse',metrics = ['mae','mse'])

print(model.summary())
  

history = model.fit(X_train,Y_train,epochs = 100,validation_split = 0.2)
preds = model.predict(x)
print(model.evaluate(X_test,Y_test))

'''



import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from keras import Sequential,layers
from keras import callbacks
class CustomStopper(keras.callbacks.EarlyStopping):
    def __init__(self): # add argument for starting epoch
        super(CustomStopper, self).__init__()
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)

'''


es = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')
'''
folds = 1
rmse_avg = []
r2_avg = []
nrmse_avg = []
loss_avg = []
count = 0
for i in tqdm(range(folds)):
	es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=0, mode='auto', restore_best_weights=True)
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
	model.compile(optimizer = 'sgd',loss = 'mse',metrics = ['mae','mse'])

	print(model.summary())


	history = model.fit(X_train,Y_train,epochs = 10,batch_size=64,validation_data = (X_test,Y_test),callbacks=[es])
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
	print("original",y[0:10])
	y1 = []
	for i in range(len(y)):
		pr = y[i] * den
		cal = pr + add
		y1.append(cal)
	print("after",y1[0:10])

	loss_avg.append(loss)
	rmse_avg.append(rmse_test)
	r2_avg.append(r2_test)
	nrmse_avg.append(nrmse)
	#print("Mean_absolute_percentage_error of test set is",mean_absolute_percentage_error(Y_test,preds))
	
	with open('res.txt','a') as f:
		count = count + 1
		f.write("fold=%s\n" % str(count))
		f.write("rmse=%s\n" % str(rmse_test))
		f.write("loss=%s\n" % str(loss))
		f.write("r score=%s\n" % str(r2_test))
		f.write("nrmse=%s\n" % str(nrmse))
		f.write("-------------------------------------------------\n")
	

	plt.plot(history.history['mae'])
	plt.plot(history.history['val_mae'])
	plt.title('model mae')
	plt.ylabel('mae')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()
	#plt.pause(3)
	#plt.close()

	plt.plot(history.history['mse'])
	plt.plot(history.history['val_mse'])
	plt.title('model mse')
	plt.ylabel('mae')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()
	# plt.pause(3)
	# plt.close()

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()
	# plt.pause(3)
	# plt.close()
	print('True 10 samples',Y_test[0:10])
	print('Predicted 10 samples',preds[0:10])
	arra = []
	arrak = []


	from tqdm import tqdm


	for i in tqdm(range(0,39988)):
	  arrak.append(i)
	print("or",preds1[0:10])
	preds11 = []
	for i in range(len(preds1)):
		pr = preds1[i] * den
		cal = pr + add
		preds11.append(cal)
	print("af",preds11[0:10])
	plt.gca().set_prop_cycle("color", ['blue','green'])
	plt.title('For DO% NN')
	plt.plot(np.array(arrak),y1)
	plt.plot(np.array(arrak),np.array(preds11))
	plt.xlabel('Index')
	plt.ylabel('Values')
	plt.legend(['Normal data', ' Predicted Output'], loc='upper left')
	plt.show()
	# plt.pause(5)
	# plt.close()
'''
nrmse_avg = np.array(nrmse_avg)
r2_avg = np.array(r2_avg)
rmse_avg = np.array(rmse_avg)
loss_avg = np.array(loss_avg)

print("RMSE is {}".format(rmse_avg.mean()))
print("R score is {}".format(r2_avg.mean()))
print("loss_avgis{}".format(loss_avg.mean()))
print("nrmse is {}".format(nrmse_avg.mean()))

'''