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


idx2char = ["e","h","i","l","o"]

# _data = np.array([["h","i","h","e","l","l","o"]]).reshape((-1.1))
_data = np.array([["h","i","h","e","l","l","o"]],dtype=np.str).reshape((-1,1))

print(_data.shape)
print(_data)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray().astype("float32") #알파벳 순서로 인코딩



print(_data)
print(_data.shape)

x_data = _data[:6,]
y_data = _data[1:,]
# y_data = np.argmax(y_data, axis=1)
print("x_data",x_data)
print("y_data",y_data)
x_data = x_data.reshape(1,6,5)
y_data = y_data.reshape(1,6,5)
print("x_data",x_data.shape)
print("y_data",y_data.shape)


num_classes = 5
batch_size = 1
sequence_length = 6

input_dim = 5
hidden_size = 5
learning_rate = 0.1


model = Sequential()

model.add(LSTM(30,input_shape=(6,5),return_sequences=True))
model.add(LSTM(10,return_sequences=True))
# model.add(Dense(100,activation="relu"))

model.add(LSTM(5,activation="softmax",return_sequences=True))



model.summary()
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


# early_stoping_callback = keras.callbacks.EarlyStopping(monitor="loss",patience=10)

history = model.fit(x_data, y_data, epochs = 600, batch_size=1, verbose=2)

print("\n test acc: %.4f"%(model.evaluate(x_data, y_data)[1]))

pre = model.predict(x_data)
print(y_data)
print(pre)
y_data = np.argmax(y_data, axis=2)
pre = np.argmax(pre, axis=2)
print(y_data)
print(pre)
result_str = [idx2char[c] for c in np.squeeze(pre)]
print("\nPrediction str : ",''.join(result_str))



