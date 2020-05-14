#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score,mean_absolute_error
from nrmse import cal_nrmse
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


df = pd.read_csv('time_data.csv')

print(df.info())

# In[34]:



from datetime import datetime, timedelta

# get current time
now = datetime.now()
d = timedelta(30)
param = now-d
print ("Today's date: ", str(now))
#add 15 days to current date
print("param",param)
lst = []
for i in range(len(df['ph'])):
    future_date_after_15_min= param + timedelta(seconds = 900)
    param = future_date_after_15_min
    lst.append(param)
    #print(now)


# In[35]:


df.head()


# In[36]:


df["Time"] = lst


# In[37]:


df.head(10)


# In[38]:


df['Time'] = pd.to_datetime(df["Time"])
df_idx = df.set_index(["Time"], drop=True)
df_idx.head(5)


# In[8]:


df_idx = df_idx.sort_index(axis=1, ascending=True)
df_idx = df_idx.iloc[::-1]


# In[39]:


df_idx.head()


# In[44]:


df_idx.iloc[int(30997)]


# In[41]:


data = df_idx[['t1']]
data.plot(y='t1')


# In[45]:



split_date = pd.Timestamp('2020-12-30 10:07:51.067414')

train = data.loc[:split_date]
test = data.loc[split_date:]

ax = train.plot(figsize=(10,12))
test.plot(ax=ax)
plt.legend(['train', 'test'])
plt.show()


# In[46]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)


# In[47]:


train_sc_df = pd.DataFrame(train_sc, columns=['Y'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Y'], index=test.index)

for s in range(1,2):
    train_sc_df['X_{}'.format(s)] = train_sc_df['Y'].shift(s)
    test_sc_df['X_{}'.format(s)] = test_sc_df['Y'].shift(s)

X_train = train_sc_df.dropna().drop('Y', axis=1)
y_train = train_sc_df.dropna().drop('X_1', axis=1)

X_test = test_sc_df.dropna().drop('Y', axis=1)
y_test = test_sc_df.dropna().drop('X_1', axis=1)

X_train = X_train.as_matrix()
y_train = y_train.as_matrix()

X_test = X_test.as_matrix()
y_test = y_test.as_matrix()


# In[23]:


print('Train size: (%d x %d)'%(X_train.shape[0], X_train.shape[1]))
print('Test size: (%d x %d)'%(X_test.shape[0], X_test.shape[1]))


# In[48]:


from sklearn.svm import SVR
regressor = SVR(kernel='rbf')


# In[49]:


regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[50]:


plt.plot(y_test)
plt.plot(y_pred)


# In[51]:


from sklearn.metrics import r2_score

def adj_r2_score(r2, n, k):
    return 1-((1-r2)*((n-1)/(n-k-1)))

r2_test = r2_score(y_test, y_pred)
print("R-squared is: %f"%r2_test)


# In[52]:


print(adj_r2_score(r2_test,39997,10000))


count = 0
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as K


# In[55]:


K.clear_session()
model = Sequential()
model.add(Dense(1, input_shape=(X_test.shape[1],), activation='tanh', kernel_initializer='lecun_uniform'))
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error',metrics=['mean_squared_error'])
model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=1)


# In[56]:


y_pred = model.predict(X_test)
print('y_pred',y_pred[:100])
print('original',y_test[:100])
plt.plot(y_test)
plt.plot(y_pred)
plt.show()
result = model.evaluate(X_test,y_test) 
loss = result[0]
print(result)
rmse_test = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test,y_pred)
print('mae',mae)
r2_test = r2_score(y_test, y_pred)
print("MSE of test set is {}".format(rmse_test))
print("R score of test set is {}".format(r2_test))
nrmse = cal_nrmse(y_test,y_pred)
print("nrmse of test set is {}".format(nrmse))
with open('res_temp_time_series.txt','a') as f:
			count = count + 1
			f.write("simple\n")
			f.write("fold=%s\n" % str(count))
			f.write("rmse=%s\n" % str(rmse_test))
			f.write("loss=%s\n" % str(loss))
			f.write("mae=%s\n"%str(mae))
			f.write("r score=%s\n" % str(r2_test))
			f.write("nrmse=%s\n" % str(nrmse))
			f.write("-------------------------------------------------\n")


# In[57]:


K.clear_session()
model = Sequential()
model.add(Dense(50, input_shape=(X_test.shape[1],), activation='tanh', kernel_initializer='lecun_uniform'))
model.add(Dense(25, input_shape=(X_test.shape[1],), activation='tanh'))
model.add(Dense(1))
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error',metrics=['mean_squared_error'])
model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=1)


# In[60]:



y_pred = model.predict(X_test)
print('y',y_pred[:100])
print('original',y_test[:100])
plt.plot(y_test)
plt.plot(y_pred)
plt.show()
r2_scor = r2_score(y_test, y_pred)
result = model.evaluate(X_test,y_test) 
loss = result[0]
print(result)
rmse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test,y_pred)
print('mae',mae)
print("MSE of test set is {}".format(rmse_test))
print("R score of test set is {}".format(r2_test))
nrmse = cal_nrmse(y_test,y_pred)
print("nrmse of test set is {}".format(nrmse))
print('adj-R-Squared: %f'%(adj_r2_score(r2_scor,39997,10000)))
print('mean-Squared_error: %f'%(mean_squared_error(y_test, y_pred)))
with open('res_temp_time_series.txt','a') as f:
			count = count + 1
			f.write("wider\n")
			f.write("fold=%s\n" % str(count))
			f.write("rmse=%s\n" % str(rmse_test))
			f.write("mae=%s\n"%str(mae))
			f.write("loss=%s\n" % str(loss))
			f.write("r score=%s\n" % str(r2_test))
			f.write("nrmse=%s\n" % str(nrmse))
			f.write("-------------------------------------------------\n")


# In[ ]:




