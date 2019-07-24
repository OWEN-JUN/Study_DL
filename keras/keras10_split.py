import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))

# #1~60
# x_train = x[:60]
# y_train = y[:60]
# #61~80
# x_val =x[60:80]
# y_val =y[60:80]
# #81~100
# x_test =x[80:]
# y_test =y[80:]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size = 0.4)



x_test, x_val, y_test, y_val = train_test_split(x_test, y_test,random_state=66, test_size = 0.5)


print(len(x_train))
print(len(x_val))
print(len(x_test))




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

model1.add(Dense(5, input_shape=(1,), activation="relu"))


model1.add(Dense(4 ))
model1.add(Dense(3))
model1.add(Dense(1, activation="relu"))



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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
def RMSE(y_test, y_):
    return np.sqrt(mean_squared_error(y_test,y_))
def RMAE(y_test, y_):
    return np.sqrt(mean_absolute_error(y_test,y_))
print("RMSE:",RMSE(y_test,y_))
print("RMAE:",RMAE(y_test,y_))