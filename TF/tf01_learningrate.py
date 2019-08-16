
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])


#모델구성
import keras
from keras.models import Sequential
from keras.layers import Dense

from time import time
from keras import layers
from keras import models

model = Sequential()

model.add(Dense(1, input_dim=1))

# model.add(Dense(3))
# model.add(Dense(4))
model.add(Dense(1))




#훈련
optimizer1 = keras.optimizers.Adam(lr=0.5)
optimizer2 = keras.optimizers.Adamax(lr=0.1)
optimizer3 = keras.optimizers.Adadelta(lr=0.3)
optimizer4 = keras.optimizers.Adagrad(lr=0.2)
optimizer5 = keras.optimizers.Nadam(lr=0.3)
optimizer6 = keras.optimizers.RMSprop(lr=0.4)
optimizer7 = keras.optimizers.SGD(lr=0.5)
keras.optimizers.TFOptimizeropti_list = [optimizer1,optimizer2,optimizer3,optimizer4,optimizer5,optimizer6,optimizer7]
opti_loss = []
predict_list = []
for i in range(7):
    model.compile(loss="mse", optimizer=opti_list[i], metrics=['mae'])
    model.fit(x,y, epochs=100, batch_size=1,verbose=2)
    loss, mse = model.evaluate(x,y, batch_size=1)
    print("mse:", loss)
    opti_loss.append(loss)
    y_= model.predict([1.5,2.5,3.5])
    predict_list.append(y_)


#평가예측
# loss, mse = model.evaluate(x,y, batch_size=1)
# print("mse:", loss)
# y_= model.predict([1.5,2.5,3.5])
# print(y_)

for i in range(len(predict_list)):
    print("optimizer:",opti_list[i])
    print("loss     :",opti_loss[i])
    print("predict  :",predict_list[i])


### RESULT
# optimizer: <keras.optimizers.Adam object at 0x000001E429B8DE80>
# loss     : 0.05409424526927372
# predict  : [[1.4475625] [2.2377734] [3.0279846]]

# optimizer: <keras.optimizers.Adamax object at 0x000001E429BB2278>
# loss     : 0.0
# predict  : [[1.5000001] [2.5      ] [3.5      ]]

# optimizer: <keras.optimizers.Adadelta object at 0x000001E429BB2F60>
# loss     : 0.0
# predict  : [[1.5000001] [2.5      ] [3.5      ]]

# optimizer: <keras.optimizers.Adagrad object at 0x000001E429BB2748>
# loss     : 0.0
# predict  : [[1.5000001] [2.5      ] [3.5      ]]

# optimizer: <keras.optimizers.Nadam object at 0x000001E429BB25F8>
# loss     : 0.0
# predict  : [[1.5000001] [2.5      ] [3.5      ]]

# optimizer: <keras.optimizers.RMSprop object at 0x000001E429BB26A0>
# loss     : 0.0
# predict  : [[1.5000001] [2.5      ] [3.5      ]]

# optimizer: <keras.optimizers.SGD object at 0x000001E429BC96D8>
# loss     : 0.0
# predict  : [[1.5000001] [2.5      ] [3.5      ]]