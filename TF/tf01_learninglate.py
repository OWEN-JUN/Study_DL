# import keras
# import matplotlib.pyplot as plt
# from keras.datasets import mnist

# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# digit = train_images[4]
# plt.figure()
# plt.imshow(digit, cmap= plt.cm.binary)
# plt.show

# print(digit)
# digit = digit.astype("float32")/255


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

model = Sequential()

model.add(Dense(5, input_dim=1, activation="relu"))
model.add(Dense(7, activation="relu"))
model.add(Dense(4))
model.add(Dense(1))


# tensorboard = TensorBoard(log_dir="./log/{}".format(time()))
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
#훈련
model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
model.fit(x,y, epochs=300, batch_size=10, callbacks=[tb_hist])


#평가예측
loss, acc = model.evaluate(x,y, batch_size=1)
print("acc:", acc)
y_= model.predict([1,2,3,4,5,6,7,8,9])
print(y_)