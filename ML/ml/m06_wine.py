import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding="utf-8")
import seaborn as sns
# plt.figure(figsize=(15,15))
# sns.pairplot(wine)
# plt.show()
y = wine["quality"]

newlist=[]
for v in list(y):
    if v<=4:
        newlist += [0]
    elif v<=7:
        newlist += [1]
    else:
        newlist += [2]

y = newlist

from keras.utils import np_utils



label = LabelEncoder()

label.fit(y)
y = label.transform(y)

y = np_utils.to_categorical(y,3)



# x = wine.drop(["quality","density",'total sulfur dioxide','chlorides',"residual sugar","fixed acidity"], axis=1)
x = wine.drop(["quality","density","fixed acidity"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)
x_test, x_val, y_test, y_val = train_test_split(x_test,y_test, test_size=0.3)


print(x.shape)

import keras
model = Sequential()
# model.add(Dense(100,input_dim=6, activation="relu"))
model.add(Dense(50,input_dim=9, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(100))

# model.add(Dropout(0.3))
# model.add(Dense(1000, activation="relu"))
# model.add(Dropout(0.3))

# model.add(Dense(200, activation="relu"))
model.add(Dense(10, activation="relu"))

model.add(Dense(3,activation="softmax"))

# model.compile(loss="mse", optimizer="adadelta", metrics=["acc"])

early = keras.callbacks.EarlyStopping(monitor="val_loss",patience=30,mode="auto")
model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["acc"])

model.fit(x_train, y_train, epochs=2000, batch_size=20, validation_data=(x_val, y_val))
print(model.predict(x_test))
print("Acc : ", model.evaluate(x_test,y_test)[1])