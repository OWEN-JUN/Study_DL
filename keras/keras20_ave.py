import keras
import plaidml.keras
plaidml.keras.install_backend()
import numpy as np
# x = np.array(range(1,101))
# y = np.array(range(1,101))
x= np.array([range(1000),range(3110,4110),range(1000)])
y= np.array([range(5010,6010)])


# print(x.shape)
# print(y.shape)
x = np.transpose(x)
y = np.transpose(y)
# print(x.shape)
# print(y.shape)


# y = y.reshape(100,2)




# print(x)

# print(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size = 0.4)




x_test, x_val, y_test, y_val = train_test_split(x_test, y_test,random_state=66, test_size = 0.5)






# x2 = np.array([4,5,6])

#모델구성
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor="lostt", patience=100)
from time import time
from keras import layers
from keras import models

model1 = Sequential()

# model1.add(Dense(40, input_dim=1, activation="relu"))
from keras import regularizers
model1.add(Dense(500, input_shape=(3,), activation="relu"))
# model1.add(BatchNormalization())
# model1.add(Dense(50,activation="relu",kernel_regularizer=regularizers.l2(0.001)))
model1.add(Dense(400,activation="relu"))
model1.add(Dense(300,activation="relu"))
model1.add(Dense(500,activation="relu"))

model1.add(Dense(20,activation="relu"))

# model1.add(Dropout(0.2))



model1.add(Dense(1))



model1.save("savetest01.h5")