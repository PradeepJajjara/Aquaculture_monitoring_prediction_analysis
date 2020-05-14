import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
data = pd.read_csv("dataset.csv")
mydata=pd.DataFrame(data)
print(mydata.head())
mydata = mydata.apply (pd.to_numeric, errors='coerce')
mydata = mydata.dropna()
print(mydata.shape)
Y=mydata['PH']
x=mydata.drop('PH',axis=1)
x=x.drop('Salinity',axis=1)
print(mydata.describe().transpose())
from sklearn  import model_selection
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=5)
phdata=mydata['PH']
temp=mydata['Temperature']
#plt.plot(temp, phdata, 'b-', label='ph Value')
#plt.plot(years, temp, 'g-', label='Temperature Value')
#plt.scatter(temp, phdata)
import seaborn as sns
#sns.pairplot(mydata)
print('x_train',x_train.shape)
print('x_test',x_test.shape)
print('y_train',y_train.shape)
print('y_test',y_test.shape)
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
dtr=DecisionTreeRegressor(random_state=1)
svmmodel=svm.SVR()
clf = GradientBoostingRegressor(**params)
lm=LinearRegression()
rdf=RandomForestRegressor()
mlp = MLPRegressor(hidden_layer_sizes=(4,4,4), activation='relu',solver='adam', max_iter=500)
lm.fit(x_train,y_train)
mlp.fit(x_train,y_train)
clf.fit(x_train,y_train)
dtr.fit(x_train,y_train)
rdf.fit(x_train,y_train)
svmmodel.fit(x_train,y_train)

from sklearn.metrics import r2_score,mean_absolute_error
print("Linear Regression Model Accuracy : %f"%(r2_score(y_test,lm.predict(x_test))))
print("Multilayer Perceptron Regression Model Accuracy : %f"%(r2_score(y_test,mlp.predict(x_test))))
print("Support Vector Machine Regression Model Accuracy : %f"%(r2_score(y_test,svmmodel.predict(x_test))))
print("Decision Tree Regression Model Accuracy : %f)"%(r2_score(y_test,dtr.predict(x_test))))
print("Gredient Boosting Regression Model Accuracy : %f)"%(r2_score(y_test,clf.predict(x_test))))
print("Random Forest Regression Model Accuracy : %f)"%(r2_score(y_test,rdf.predict(x_test))))

print("All Model's Mean Squared Error ")
print("Linear Regression Model MSE Error : %f"%(mean_squared_error(y_test,lm.predict(x_test))))
print("Multilayer Perceptron Regression Model MSE Error : %f"%(mean_squared_error(y_test,mlp.predict(x_test))))
print("Support Vector Machine Regression Model MSE Error : %f"%(mean_squared_error(y_test,svmmodel.predict(x_test))))
print("Decision Tree Regression Model MSE Error: %f)"%(mean_squared_error(y_test,dtr.predict(x_test))))
print("Gredient Boosting Regression Model MSE Error : %f)"%(mean_squared_error(y_test,clf.predict(x_test))))
print("Random forest Regression Model MSE Error : %f)"%(mean_squared_error(y_test,rdf.predict(x_test))))

print("RMSE")
print("Linear Regression: %f"%(np.sqrt(mean_squared_error(y_test,lm.predict(x_test)))))
print("Neural Network: %f"%(np.sqrt(mean_squared_error(y_test,mlp.predict(x_test)))))
print("Support Vector Machine: %f"%(np.sqrt(mean_squared_error(y_test,svmmodel.predict(x_test)))))
print("Decision Tree: %f"%(np.sqrt(mean_squared_error(y_test,dtr.predict(x_test)))))
print("Gradient Boosting: %f"%(np.sqrt(mean_squared_error(y_test,clf.predict(x_test)))))
print("Random Forest: %f"%(np.sqrt(mean_squared_error(y_test,rdf.predict(x_test)))))

print("MAE")
print("Random forest Regression Model MSE Error : %f)"%(mean_absolute_error(y_test,rdf.predict(x_test))))
print("LR : %f)"%(mean_absolute_error(y_test,lm.predict(x_test))))
print("Gradient Boosting: %f)"%(mean_absolute_error(y_test,clf.predict(x_test))))
print("Decision Tree : %f)"%(mean_absolute_error(y_test,dtr.predict(x_test))))
print("Support Vector : %f)"%(mean_absolute_error(y_test,svmmodel.predict(x_test))))
print("Neural Network : %f)"%(mean_absolute_error(y_test,mlp.predict(x_test))))
'''
sns.relplot(x="PH", y="Turbidity", kind="line", data=mydata)

sns.relplot(x="PH", y="Temperature", kind="line", data=mydata)
sns.relplot(x="PH", y="DO%", kind="line", data=mydata)
sns.relplot(data=mydata)
sns.lineplot(data=mydata)
sns.scatterplot(data=mydata)
'''

models = []
models.append(('LR', lm))
models.append(('MLP',mlp))
models.append(('SVM', svmmodel))
models.append(('DTR', dtr))
models.append(('GBR', clf))
models.append(('RF', rdf))
# evaluate each model in turn
results = []
names = []

for name, model in models:
 kfold = model_selection.KFold(n_splits=10, random_state=5)
 cv_results =r2_score(y_test,model.predict(x_test))
 results.append(cv_results)
 names.append(name)
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)
# boxplot algorithm comparison
print(results)
plt.show()	
ax=sns.barplot(names,results).set_title("Model Comparision")





