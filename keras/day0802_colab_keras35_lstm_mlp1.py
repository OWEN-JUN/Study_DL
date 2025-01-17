# -*- coding: utf-8 -*-
"""day0802_colab_keras35_lstm_mlp1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jHYOiQ4NHbAlSm2lZpVP8qwXACk67x3y
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
# from sklearn import standard



x = np.array(range(1,101))
print(x)

size = 8

def split_5(seq, size):


    aaa=[]
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append(subset)

    print(type(aaa))
    return np.array(aaa)


dataset=split_5(x, size)
print("==================")


print(dataset)
x = dataset[:,:-4]
y = dataset[:,-4:]
print(x)
print(x.shape)
print(y)
print(y.shape)

x = x.reshape(x.shape[0],x.shape[1],1)
print(x.shape)


# x_test = np.array([[[11],
#   [12],
#   [13],
#   [14]],

#  [[12],
#   [13],
#   [14],
#   [15]],

#  [[13],
#   [14],
#   [15],
#   [16]],

#  [[14],
#   [15],
#   [16],
#   [17]],

#  [[15],
#   [16],
#   [17],
#   [18]],

#  [[16],
#   [17],
#   [18],
#   [19]]])

# y_test = np.array([15, 16, 17, 18, 19, 20])

# print(x.shape)
# print(y.shape)
# print(x_test.shape)
# print(y_test.shape)


x_test = x
y_test = y
#모델구성

model = Sequential()

model.add(LSTM(32, input_shape=(4,1),return_sequences=True))


model.add(LSTM(10))

model.add(Dense(5, activation="relu"))
model.add(Dense(3))
model.add(Dense(4))
model.summary()
model.compile(loss = "mse",optimizer="adam", metrics=['accuracy'])
from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor="loss", patience=300, mode="auto")
model.fit(x,y,epochs=3000, callbacks=[early])

loss, acc = model.evaluate(x_test, y_test)
y_ = model.predict(x_test)
print(y_, "ori: ",y_test)
print(loss, acc)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
def RMSE(y_test, y_):
    return np.sqrt(mean_squared_error(y_test,y_))
def RMAE(y_test, y_):
    return np.sqrt(mean_absolute_error(y_test,y_))
print("RMSE:",RMSE(y_test,y_))
print("RMAE:",RMAE(y_test,y_))
print("r2:",r2_score(y_test,y_))

