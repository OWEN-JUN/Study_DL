import numpy as np
# x = np.array(range(1,101))
# y = np.array(range(1,101))
x= np.array([range(100),range(311,411),range(1001,1101)])
y= np.array([range(501,601),range(711,811),range(1501,1601)])
x = np.transpose(x)
y = np.transpose(y)
# y = y.reshape(100,2)

print(x.shape)
print(x)
print(y.shape)
print(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size = 0.4)



x_test, x_val, y_test, y_val = train_test_split(x_test, y_test,random_state=66, test_size = 0.5)






# x2 = np.array([4,5,6])

#모델구성
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input

from time import time
from keras import layers
from keras import models



# model1.add(Dense(40, input_dim=1, activation="relu"))
input1 = Input(shape=(3,))
dense1 = Dense(100, activation="relu")(input1)
dense1_1 = Dense(30, activation="relu")(dense1)
dense1_2= Dense(3, activation="relu")(dense1_1)


model=Model(input=input1, output = dense1_2)
model.summary()


# # tensorboard = TensorBoard(log_dir="./log/{}".format(time()))
# # tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
# #훈련

model.compile(loss="MSE", optimizer="adam", metrics=['accuracy'])
model.fit(x_train,y_train, epochs=500, batch_size=1, validation_data=(x_val,y_val))
# # # model1.fit(x_train,y_train, epochs=100 )





# # #평가예측
print("model1")
loss, acc = model.evaluate(x_test,y_test, batch_size=1)
print("acc:",acc)
y_= model.predict(x_test)
print(y_)

# model1.summary()


# # #RMSE 구하기
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
def RMSE(y_test, y_):
    return np.sqrt(mean_squared_error(y_test,y_))
def RMAE(y_test, y_):
    return np.sqrt(mean_absolute_error(y_test,y_))
print("RMSE:",RMSE(y_test,y_))
print("RMAE:",RMAE(y_test,y_))
print("r2:",r2_score(y_test,y_))
