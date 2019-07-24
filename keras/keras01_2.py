import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])


#모델구성
import keras
from keras.models import Sequential
from keras.layers import Dense

from time import time
from keras import layers
from keras import models

model1 = Sequential()

model1.add(Dense(40, input_dim=1, activation="relu"))
model1.add(Dense(30, activation="relu"))
model1.add(Dense(10))
model1.add(Dense(1))



# tensorboard = TensorBoard(log_dir="./log/{}".format(time()))
# tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
#훈련

model1.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
model1.fit(x,y, epochs=100, batch_size=1, )




#평가예측
print("model1")
loss, acc = model1.evaluate(x,y, batch_size=1)

y_= model1.predict([1,2,3,4,5,6,7,8,9])
print(y_)