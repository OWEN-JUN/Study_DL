from sklearn.linear_model import LinearRegression



import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



df = pd.read_csv("./data/tem10y.csv", encoding="utf-8")
# train_year = (df["연"] <= 2015)
# test_year = (df["연"] >= 2016)
interval=6

def make_data(data):
    x=[]
    y=[]
    temps=list(data["기온"])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa=[]
        for p in range(interval):
            d = i+p-interval
            xa.append(temps[d])
        x.append(xa)
    return (x,y)
from sklearn.model_selection import train_test_split
x, y = make_data(df)
x, y = np.array(x), np.array(y)

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.15)


# test_x, test_y = make_data(df[test_year])

print(np.array(train_x).shape)
print(np.array(train_y).shape)
print(train_y)

train_x = np.array(train_x)
train_x = train_x.reshape((-1,interval,1))
train_y = np.array(train_y)
test_x = np.array(test_x)
test_x = test_x.reshape((-1,interval,1))
test_y = np.array(test_y)
# import sys
# sys.exit()
model = Sequential()
import keras

# model.add(TimeDistributed(Dense(interval, input_dim=interval,activation="relu")))
# model.add(Dense(60,input_dim=6,))
# model.add(Dense(10))
# model.add(TimeDistributed(Dense(10)))
from keras import regularizers
model.add(LSTM(64, input_shape=(interval,1),return_sequences=True))
model.add(LSTM(16))
model.add(Dense(32,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
# model.add(Dense(10,activation="relu"))
# keras.optimizers.adam(lr = 0.001,)
model.add(Dense(1))
model.compile(optimizer='RMSprop',
              loss='mse',
              metrics=['mae'])

early = keras.callbacks.EarlyStopping(monitor="loss", patience=30, mode="auto")
model.fit(train_x, train_y,epochs=300, batch_size=20, callbacks=[early])

# pre_y = lr.predict(test_x)
pre_y = model.predict(test_x)

from sklearn.metrics import r2_score
r2 = r2_score(test_y, pre_y)
print("r2:",r2)
print("mse mae:",model.evaluate(test_x, test_y))
# print(lr.score(test_x, test_y))



plt.figure(figsize=(10,6), dpi=100)
plt.plot(test_y, c="r")
plt.plot(pre_y, c="b")
plt.savefig('tenki-kion-lr.png')
plt.show()