from numpy import array
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
#데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12]])

y = array([4,5,6,7,8,9,10,11,12,13])

print("x.shape:",x.shape)
print("y.shape",y.shape)


x = x.reshape((x.shape[0],x.shape[1],1))
print("x.shape",x.shape)

#2모델구성

model = Sequential()

model.add(LSTM(10,activation="relu", input_shape=(3,1)))
model.add(Dense(5))
model.add(Dense(1))


model.summary()
model.compile(loss="mse",optimizer="adam")
model.fit(x, y, epochs=500, batch_size=3)

x_input = array([11,12,13])

x_input = x_input.reshape((1,3,1))
print(x_input)

yhat = model.predict(x_input)
print(yhat)
model.summary()
