from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from keras.models import *
from keras.layers import *
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


seed =0
np.random.seed(seed)
tf.set_random_seed(seed)
def minmax_scaler(x,a):
    scaler =MinMaxScaler()
    scaler.fit(x[:,a:a+1])    
    x[:,a:a+1] = scaler.transform(x[:,a:a+1])
    return x

dataset =np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
x=dataset[:,:-1]
y=dataset[:,-1]


# x = minmax_scaler(x,1)
# x = minmax_scaler(x,2)
# x = minmax_scaler(x,3)

import keras
model = Sequential()
model.add(Dense(100,input_dim=8, activation="relu"))
model.add(Dense(20, activation="relu"))
model.add(Dense(20, activation="relu"))
model.add(Dense(20, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
sig = np.float(0.6)
model.fit(x, y, epochs=300, batch_size=20)

print("Acc : ", model.evaluate(x,y)[1])
print("predict: \n", model.predict_classes(x))
