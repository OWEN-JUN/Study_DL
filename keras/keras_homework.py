import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))



x_train = np.array([i for i in range(1,101)])
y_train = np.array([i for i in range(501,601)])
x_test = np.array([i for i in range(1001,1101)])
y_test = np.array([i for i in range(1101,1201)])





# print(x_train)
# print(y_train)
# x2 = np.array([4,5,6])

# #모델구성
import keras
from keras.models import Sequential
from keras.layers import Dense

from time import time
from keras import layers
from keras import models

model1 = Sequential()

model1.add(Dense(500, input_dim=1, activation="relu"))

model1.add(Dense(4000))
model1.add(Dense(2000))

model1.add(Dense(300))
model1.add(Dense(100))
model1.add(Dense(1))



# model1.summary()


# # tensorboard = TensorBoard(log_dir="./log/{}".format(time()))
# # tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
# #훈련

model1.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
model1.fit(x_train,y_train, epochs=800, batch_size=10)
# model1.fit(x_train,y_train, epochs=100 )





# #평가예측
print("model1")

loss, acc = model1.evaluate(x_test,y_test, batch_size=3)
print("acc:",acc)
y_= model1.predict(x_test)
print(y_)

model1.summary()
print("acc:",acc)


print("acc:",acc)

print(model1.layers[-1].get_weights())