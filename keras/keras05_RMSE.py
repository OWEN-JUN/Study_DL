import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
x_val = np.array([101,102,103,104,105])
y_val = np.array([101,102,103,104,105])




# x3 = np.array([101,102,103,104,105,106])
# x4 = np.array(range(30,50))
# print(x4)

# x2 = np.array([4,5,6])

#모델구성
import keras
from keras.models import Sequential
from keras.layers import Dense

from time import time
from keras import layers
from keras import models

model1 = Sequential()

# model1.add(Dense(40, input_dim=1, activation="relu"))

model1.add(Dense(25, input_shape=(1,),activation="relu"))
model1.add(Dense(5, ))

model1.add(Dense(1,))





# model1.add(Dense(3))

# model1.add(Dense(2))
# model1.add(Dense(1, activation="relu"))



model1.summary()


# tensorboard = TensorBoard(log_dir="./log/{}".format(time()))
# tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
#훈련

# model1.compile(loss="MSE", optimizer="adam", metrics=['accuracy'])
# model1.compile(loss="MSE", optimizer="adam", metrics=['MSE'])
model1.compile(loss="MSE", optimizer="adam", metrics=['accuracy'])
model1.fit(x_train,y_train, epochs=500, batch_size=1, validation_data=(x_val, y_val))
# model1.fit(x_train,y_train, epochs=100 )





#평가예측
print("model1")
loss, acc = model1.evaluate(x_test,y_test, batch_size=1)
print("acc:",acc)
y_= model1.predict(x_test)
print(y_)

model1.summary()


#RMSE 구하기
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
def RMSE(y_test, y_):
    return np.sqrt(mean_squared_error(y_test,y_))
def RMAE(y_test, y_):
    return np.sqrt(mean_absolute_error(y_test,y_))
print("RMSE:",RMSE(y_test,y_))
print("RMAE:",RMAE(y_test,y_))
print("r2:", r2_score(y_test,y_))