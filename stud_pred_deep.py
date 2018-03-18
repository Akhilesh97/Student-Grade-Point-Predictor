# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 01:32:14 2017

@author: karthik
"""

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('example1.csv')
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()

X = dataset.iloc[:,:11].values
y = dataset.iloc[:, 11].values
X=sc.fit_transform(X)
y=sc.fit_transform(y)
X=np.reshape(X,(2000,1,11))

# Encoding categorical data
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
'''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''
X_train=X[:1600,:,:]
X_test=X[1600:,:,:]
y_train=y[:1600]
y_test=y[1600:]
y_test1=y[:1600]
# Feature Scaling
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Dropout,Bidirectional

regressor=Sequential()
regressor.add(LSTM(units=20,activation='sigmoid',input_shape=(None,11),return_sequences=  True))
regressor.add(Bidirectional(LSTM(units=20,activation='sigmoid',return_sequences=True)))
#regressor.add(Dropout(0.20))
regressor.add(Bidirectional(LSTM(units=20,activation='sigmoid',return_sequences=True)))
regressor.add(Bidirectional(LSTM(units=20,activation='sigmoid',return_sequences=True)))
regressor.add(Bidirectional(LSTM(units=20,activation='sigmoid')))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam',loss='mean_squared_error')
regressor.fit(X_train,y_train,batch_size=32,epochs=400)
predicted=regressor.predict(X_train)
predicted=sc.inverse_transform(predicted)
y_test=sc.inverse_transform(y_test1)
from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error( predicted,y_test1)**0.5
c1=0
c2=0
c3=0
for i in range(len(predicted)):
    err=(abs(predicted[i]-y_test[i])*100)/y_test[i];
    if err<=5:
        c1=c1+1
    elif err<=10:
        c2=c2+1
    elif err<=20:
        c3=c3+1
print("no of student marks predicted with less than or equal to 5% error",c1)
print("no of student marks predicted with less than or equal to 10% error",c2)
print("no of student marks predicted with less than or equal to 20% error",c3)        
        
    
    
'''compare y_test and predicted to get an underdtand the prediction'''