import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


from keras.utils import np_utils
zoo = pd.read_csv("./Study_DL/TF/data/data-04-zoo.csv")

zoo = np.array(zoo)
x_train = zoo[:,:-1]
y_train = zoo[:,-1]
y_train = np_utils.to_categorical(y_train,7)








import keras

model = Sequential()
model.add(Dense(7,input_dim=16, activation="softmax"))
# model.add(Dense(50,input_dim=9, activation="relu"))
# model.add(Dense(10, activation="relu"))
# model.add(Dense(100))

# model.add(Dropout(0.3))
# model.add(Dense(1000, activation="relu"))
# model.add(Dropout(0.3))

# model.add(Dense(200, activation="relu"))
# model.add(Dense(10, activation="relu"))

# model.add(Dense(7,activation ="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["acc"])

early = keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,mode="auto")

model.fit(x_train, y_train, epochs=500, batch_size=20)
for i in range(len(y_train)):
    print("predict : ",np.argmax(model.predict(x_train),1)[i],"true : ",np.argmax(y_train,1)[i])
print("Acc : ", model.evaluate(x_train,y_train)[1])