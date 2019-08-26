import tensorflow as tf
import matplotlib.pyplot as plt
import random
from keras.models import * #sequential
from keras.layers import * #dense, dropout, flatten, conv2d, maxpooling2d
from keras.callbacks import * #modelcheckpoint, earlystopping
from keras.utils import *
import matplotlib.pyplot as plt
import keras
import os

from keras.datasets import mnist

tf.set_random_seed(777)



(x_train, y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],28,28).astype("float32")/255
x_test = x_test.reshape(x_test.shape[0],28,28).astype("float32")/255
y_train= np_utils.to_categorical(y_train)
y_test= np_utils.to_categorical(y_test)


model = Sequential()

model.add(LSTM(30,input_shape=(28,28),return_sequences=True))
model.add(LSTM(10))
model.add(Dense(100,activation="relu"))

model.add(Dense(10,activation="softmax"))




model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


early_stoping_callback = keras.callbacks.EarlyStopping(monitor="loss",patience=5)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = 100, batch_size=500, verbose=2,callbacks=[early_stoping_callback])

print("\n test acc: %.4f"%(model.evaluate(x_test, y_test)[1]))









