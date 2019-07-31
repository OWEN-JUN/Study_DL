
import numpy as np

import keras

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import *
import matplotlib.pyplot as plt



a = np.array(range(1,101))
batch_size = 2
size = 5

def split_5(seq, size):


    aaa=[]
    for i in range(len(a)-size+1):
        subset = a[i:(i+size)]
        aaa.append(subset)

    print(type(aaa))
    return np.array(aaa)

dataset = split_5(a, size)
print("_____________________________")
print(dataset)
print(dataset.shape)


x_train = dataset[:,0:4]
x_train = x_train.reshape(x_train.shape[0],-1,1)
y_train = dataset[:,4]
print(y_train)
x_test = x_train +100 
y_test = y_train +100




model = Sequential()
model.add(LSTM(10, batch_input_shape=(batch_size,4,1),stateful=True)) #상태유지 stateful
model.add(BatchNormalization())
model.add(Dense(4,activation="relu"))
# model.add(Dropout(0.2))
model.add(Dense(2,activation="relu"))
# model.add(Dense(1,activation="relu"))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())

# model.add(LSTM(128,activation="relu"))
# model.add(Dense(1))
model.add(Dense(1))


hist_loss = []
hist_acc = []
hist_val_loss = []

model.summary()
model.compile(loss="mse", optimizer="adam", metrics=["mse"])

# tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor="val_loss",patience = 10, mode="auto")


num_epochs = 300
for epoch_idx in range(num_epochs):
  print("epochs:"+str(epoch_idx))
  history=model.fit(x_train, y_train, epochs=20, batch_size=batch_size, verbose=2, shuffle=False, validation_data=(x_test, y_test),callbacks=[early_stopping])
  hist_loss.append(history.history["mean_squared_error"][0])
  hist_acc.append(history.history["val_mean_squared_error"])
  
#   print(hist_loss)
#   print(hist_acc)
#   mse1, acc1= model.evaluate(x_train, y_train, batch_size=1)
  model.reset_states()
#   mse2, acc2= model.evaluate(x_train, y_train, batch_size=1)
#   print(mse1, acc1)
#   print(mse2, acc2)

mse, _= model.evaluate(x_train, y_train, batch_size=batch_size)
print("mse:",mse)
model.reset_states()
y_=model.predict(x_test, batch_size=batch_size)

print(y_[0:10])



from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
def RMSE(y_test, y_):
    return np.sqrt(mean_squared_error(y_test,y_))
def RMAE(y_test, y_):
    return np.sqrt(mean_absolute_error(y_test,y_))
print("RMSE:",RMSE(y_test,y_))
print("RMAE:",RMAE(y_test,y_))
print(hist_loss)
hist_loss = np.array(hist_loss).flatten()
# hist_acc = np.array(hist_acc).flatten()


plt.plot(hist_loss)
plt.title("model accuracy")
plt.ylabel("mse")
plt.xlabel("epoch")
plt.legend(["train","test"], loc="upper left")
plt.show()


#mse <= 1
#rmse
#r2
#earlystop
#tensorboard
#matplotlob :mse/epochs