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
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers.merge import concatenate

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


merge1 = concatenate([dense1_3, dense2_3])
middle1 = Dense(10)(merge1)
middle2 = Dense(5)(middle1)
middle3 = Dense(7)(middle2)
##################################output_model#######################3


output1 = Dense(30)(middle3)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(20)(middle3)
output2_2 = Dense(3)(output2)


model = Model(inputs = [input1, input2], outputs=[output1_3, output2_2])



model.summary()


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

model.compile(loss="MSE", optimizer="adam", metrics=['accuracy'])
model.fit([x1_train,x2_train],[y1_train,y2_train], epochs=100, batch_size=1, validation_data=([x1_val,x2_val],[y1_val,y2_val]))
# # # model1.fit(x_train,y_train, epochs=100 )





# # #평가예측
print("model1")
l1,l2,l3,acc1,acc2 = model.evaluate([x1_test,x2_test],[y1_test,y2_test], batch_size=1)
print(l1,": ",l2,": ",l3)
print("acc1:",acc1)
print("acc2:",acc2)
y1_, y2_= model.predict([x1_test,x2_test])
print(y1_)
print(y2_)

# model1.summary()


# # #RMSE 구하기
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
# [y1_test,y2_test], [y1_,y2_]
# def RMSE(y1_test,y2_test, y1_,y2_):
def RMSE(y_test,y_):
    # return np.sqrt(mean_squared_error(y1_test,y1_)),np.sqrt(mean_squared_error(y2_test,y2_))
    return np.sqrt(mean_squared_error(y_test,y_))

# def RMAE(y1_test,y2_test, y1_,y2_):
def RMAE(y_test,y_):
    # return np.sqrt(mean_absolute_error(y1_test,y1_)),np.sqrt(mean_absolute_error(y2_test,y2_))
    return np.sqrt(mean_absolute_error(y_test,y_))


print(np.array([y1_test,y2_test]).reshape(-1,1))
# print("RMSE:",RMSE(y1_test,y2_test, y1_,y2_))
print("RMSE:",RMSE(np.array([y1_test,y2_test]).reshape(-1,1), np.array([y1_,y2_]).reshape(-1,1)))
# print([y1_test,y2_test])
# print("RMAE:",RMAE(y1_test,y2_test, y1_,y2_))
print("RMAE:",RMAE(np.array([y1_test,y2_test]).reshape(-1,1), np.array([y1_,y2_]).reshape(-1,1)))
# print("r2:",r2_score(y1_test, y1_))
# print("r2:",r2_score(y2_test, y2_))

print(r2_score(np.array([y1_test,y2_test]).reshape(-1,1), np.array([y1_,y2_]).reshape(-1,1)))