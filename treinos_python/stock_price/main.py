# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:34:19 2020

@author: apdde
"""

"""
Description : This program uses an artificial recurent neural network called Long Short Term Memory (LSTM)
to predict the closing stock price of a corporation (ic. Petrobras) using the past years stock price
"""

#Import libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Get the stock quote
df = web.DataReader('PETR4.SA', data_source='yahoo', start='2012-01-01', end='2020-04-09')
#Show the data
#df

#Get the number of rows and columns
#df.shape

#Visualize the closing price history

plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price BR($)', fontsize=18)
plt.show()


# Create a new dataframe with only the close column
data = df.filter(['Close'])
#Convert the dataframe to a numpyarray
dataset = data.values

#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


#Create the training dataset

#Create the scaled training dataset
train_data = scaled_data[0:training_data_len, :]
#Split the data into x_train and y_trin
x_train = [] #independent variables
y_train = [] #target variables

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
#    if i <= 61:
#        print(x_train)
#        print(y_train)
#        print()

#Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaope the data

#LSTM expects to receive a dataset in a 3D format

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build the LSTM model

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model

model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


#Creating the test dataset

#Create a new array containing scaled values from index 1577 (80 pct of the length) to 2046 (total lenght)
test_data = scaled_data[training_data_len - 60:, :]

#Create the datasets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


#Convert the data to a numpy array
x_test = np.array(x_test)

#Reshape the data (LSTM)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

    
##Get the root mean squared error (RMSE) [evalueate the accuracy of the model]
"""
#You want the square error to be as close to zero as possible. 
That indicates that: The rate of error between the model and the actual values in the source
is small, meaning that the model is making good predictions
"""

rsme = np.sqrt(np.mean(predictions - y_test) ** 2)
#print value to check

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visualize the model
plt.figure(figsize=(16, 8))
plt.title('Model Evaluation')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Cole Price BR$', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#Show the actual price and predicted prices

#type valid


#Get the quote

stock_quote = web.DataReader('PETR4.SA', data_source='yahoo', start='2012-01-01', end='2020-04-09')

#Create a new dataframe
new_df = stock_quote.filter(['Close'])
#get the last 60 day closing values and convert dataframe to array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_sixty_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#A´´end the past 60 days to the X_test
X_test.append(last_sixty_days_scaled)
#Convert the X_test dataset to a numpy array
X_test = np.array(X_test)
#Reshape the data to be 3D
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)

#print the predicted price (pred_price) [closing] for the following day, according to the model


#Check the actual price by the day with this line
stock_quote2 = web.DataReader('PETR4.SA', data_source='yahoo', start='2020-04-09', end='2020-04-09') 
#Set the start and end date to the desired date
