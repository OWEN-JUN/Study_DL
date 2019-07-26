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
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor="lostt", patience=100)
from time import time
from keras import layers
from keras import models

# model1 = Sequential()

# # model1.add(Dense(40, input_dim=1, activation="relu"))
# from keras import regularizers
# model1.add(Dense(100, input_shape=(3,), activation="relu", kernel_regularizer=regularizers.l1(0.01)))
# # model1.add(BatchNormalization())
# # model1.add(Dense(50,activation="relu",kernel_regularizer=regularizers.l2(0.001)))
# model1.add(Dense(30,activation="relu"))

# model1.add(Dense(20,activation="relu"))

# # model1.add(Dropout(0.2))



# model1.add(Dense(1))



# model1.summary()

model1 = load_model("savetest01.h5")
# # tensorboard = TensorBoard(log_dir="./log/{}".format(time()))
# write_graph=True,
#     write_grads=False,
#     write_images=False,
#     embeddings_freq=0,
#     embeddings_layer_names=None,
#     embeddings_metadata=None,
#     embeddings_data=None,
#     update_freq='epoch',
#     profile_batch=2
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
# #훈련

model1.compile(loss="MSE", optimizer="adam", metrics=['accuracy'])
model1.fit(x_train,y_train, epochs=300,  batch_size=60,validation_data=(x_val,y_val),verbose=2,callbacks=[early, tb_hist])
# # # model1.fit(x_train,y_train, epochs=100 )





# # #평가예측
print("model1")
loss, acc = model1.evaluate(x_test,y_test, batch_size=1)
print(loss)
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
print(loss)
