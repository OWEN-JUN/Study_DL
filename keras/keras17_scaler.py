import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler, MinMaxScaler

stand = MinMaxScaler()

a = np.array(range(11,21))

size = 5

def split_5(seq, size):


    aaa=[]
    for i in range(len(a)-size+1):
        subset = a[i:(i+size)]
        aaa.append(subset)

    print(type(aaa))
    return np.array(aaa)


dataset=split_5(a, size)
print("==================")


print(dataset)
x = dataset[:,:-1]
y = dataset[:,-1]
print(x)
print(x.shape)
print(y)
print(y.shape)
stand.fit(x)
x_stand = stand.transform(x)
x_stand = x_stand.reshape(x_stand.shape[0],x_stand.shape[1],1)
# print(x.shape)
# print(x)
# print(y)

x_test = np.array([[[11],
  [12],
  [13],
  [14]],

 [[12],
  [13],
  [14],
  [15]],

 [[13],
  [14],
  [15],
  [16]],

 [[14],
  [15],
  [16],
  [17]],

 [[15],
  [16],
  [17],
  [18]],

 [[16],
  [17],
  [18],
  [19]]])

y_test = np.array([15, 16, 17, 18, 19, 20])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1])

print(x_test)
x_test_stand = stand.transform(x_test)
x_test_stand = x_test_stand.reshape(x_test_stand.shape[0],x_test_stand.shape[1],1)
print(x.shape)
print(y.shape)
print(x_test.shape)
print(y_test.shape)



#모델구성

model = Sequential()

model.add(LSTM(32, input_shape=(4,1),return_sequences=True))


model.add(LSTM(10))

model.add(Dense(500, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(30, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(8))
model.add(Dense(1))


model.summary()
model.compile(loss = "mse",optimizer="adam", metrics=['accuracy'])
from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor="loss", patience=100, mode="auto")
model.fit(x_stand,y,epochs=3000,callbacks=[early])

loss, acc = model.evaluate(x_test_stand, y_test)
y_ = model.predict(x_test_stand)
print(y_, "ori: ",y_test)
print(loss, acc)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
def RMSE(y_test, y_):
    return np.sqrt(mean_squared_error(y_test,y_))
def RMAE(y_test, y_):
    return np.sqrt(mean_absolute_error(y_test,y_))
print("RMSE:",RMSE(y_test,y_))
print("RMAE:",RMAE(y_test,y_))
print("r2:",r2_score(y_test,y_))

