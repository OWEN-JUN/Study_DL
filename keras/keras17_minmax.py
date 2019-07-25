from numpy import array
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
#데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
# minmax = MinMaxScaler()
scaler.fit(x)
# minmax.fit(x)
x = scaler.transform(x)
# x = minmax.transform(x)
print(x)

print("x.shape:",x.shape)
print("y.shape",y.shape)
from sklearn.model_selection import train_test_split
x, x1, y, y1 = train_test_split(x, y, random_state=66, test_size = 0.1)

x = np.r_[x,x1]
y = np.r_[y,y1]

x = np.r_[x,x]
y = np.r_[y,y]



x = x.reshape((x.shape[0],x.shape[1],1))
print("x.shape",x.shape)

#2모델구성

model = Sequential()

model.add(LSTM(20,activation="sigmoid", input_shape=(3,1)))
model.add(Dense(10))
model.add(Dense(100))

model.add(Dense(5))

model.add(Dense(1))


model.summary()
model.compile(loss="mse",optimizer="adam")
model.fit(x, y, epochs=2000, batch_size=10)

x_input = array([[11,12,13]])
x_input = scaler.transform(x_input)
x_input = x_input.reshape((1,3,1))
print(x_input)

yhat = model.predict(x_input)
print(yhat)
x_input = array([[25,35,45]])
x_input = scaler.transform(x_input)
x_input = x_input.reshape((1,3,1))
x_input = x_input.reshape((1,3,1))
print(x_input)

yhat = model.predict(x_input)
print(yhat)
