import numpy as np

x1= np.array([range(100),range(311,411),range(100)])
y1= np.array([range(501,601),range(711,811),range(100)])
x2= np.array([range(100,200),range(311,411),range(100,200)])
y2= np.array([range(501,601),range(711,811),range(100)])

# print(x1.shape)
# print(x1)
# print(y1.shape)
# print(y1)

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)


print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)





from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=66, test_size = 0.4)

x1_test, x1_val, y1_test, y1_val = train_test_split(x1_test, y1_test,random_state=66, test_size = 0.5)

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=66, test_size = 0.4)

x2_test, x2_val, y2_test, y2_val = train_test_split(x2_test, y2_test,random_state=66, test_size = 0.5)



print(x2_test.shape)




# x2 = np.array([4,5,6])

#모델구성
import keras
from keras.models import Sequential
from keras.layers import Dense, Input

from time import time
from keras import layers
from keras import models

# model1 = Sequential()

input1 = Input(shape=(3,))
dense1 = Dense(100, activation = "relu")(input1)
dense1_2 = Dense(30)(dense1)
dense1_3 = Dense(7)(dense1_2)

input2 = Input(shape=(3,))
dense2 = Dense(50, activation = "relu")(input2)
dense2_2 = Dense(30)(dense2)
dense2_3 = Dense(7)(dense2_2)



# model1.add(Dense(5, input_shape=(3,), activation="relu"))


# model1.add(Dense(6 ))
# model1.add(Dense(6))

# model1.add(Dense(6))
# model1.add(Dense(6))
# model1.add(Dense(6))


# model1.add(Dense(3))



# model1.summary()


# # tensorboard = TensorBoard(log_dir="./log/{}".format(time()))
# # tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
# #훈련

model1.compile(loss="MSE", optimizer="adam", metrics=['accuracy'])
model1.fit(x_train,y_train, epochs=500, batch_size=1, validation_data=(x_val,y_val))
# # # model1.fit(x_train,y_train, epochs=100 )





# # #평가예측
print("model1")
loss, acc = model1.evaluate(x_test,y_test, batch_size=1)
print("acc:",acc)
y_= model1.predict(x_test)
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
