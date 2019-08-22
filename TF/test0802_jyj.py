import pandas as pd

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
test = pd.read_csv("./data/test0822.csv")
test_columns = test.columns.drop("date")
# print(test_columns)
# print(test)


max = test.max() ##max = 9
min = test.min() ##min = 0

# print(max,min)

test["kp_sum"] =test[test_columns].sum(axis=1)
test["k_out"] = 0


# print(test)
# print(len(test))
# test_date_count = test["date"].count()
# test_count = test[test["kp_sum"]>50]["kp_sum"].count()
# print("total_cnt:",test_date_count)
# print("count:",test_count)

# day = test[test["date"]=="2007-07-11"]
# day2 = test[test["date"]=="2007-07-15"]
train1 = test[:3113]
train2 = test[3118:]
l1 = test[3113:3118]
# train = test[]

stan = StandardScaler()

train1_sum_list = np.array(train1["kp_sum"]).reshape((-1,1))
train2_sum_list = np.array(train2["kp_sum"]).reshape((-1,1))
stan.fit(train1_sum_list)
train1["k_out"] = stan.transform(train1_sum_list)
train2["k_out"] = stan.transform(train2_sum_list)
train1 = train1.drop(["date","kp_sum"],axis=1)
train2 = train2.drop(["date","kp_sum"],axis=1)
train1_c = np.array(train1[test_columns]).reshape((-1,1))
# stan.fit(train1_c)

# for i in test_columns:
#     train1_c = np.array(train1[i]).reshape((-1,1))
#     train2_c = np.array(train2[i]).reshape((-1,1))

    
#     train1[i] = stan.transform(train1_c)
#     train2[i] = stan.transform(train2_c)

train1_list = np.array(train1)
train2_list = np.array(train2)
print(len(train1))
print(len(train2))
print(train1_list.shape)
print(train2.shape)
print(l1)


###예측에 쓰이는 기간

def make_set(seq,size = 2,pre_day = 5):
    test_up = up_split(seq,size,pre_day)
    test_down = down_split(seq,size,pre_day)
    test_label = split_label(seq,size,pre_day)
    return test_up, test_down, test_label

##TRAIN 데이터와 LABEL데이터를 나누는 곳
def up_split(seq,size,pre_day):
    up = []
    for i in range((len(seq)-size-pre_day)-size+1):
        subset = seq[i:(i+size)]
        up.append(subset)
    print(type(up))
    return np.array(up)

def down_split(seq,size,pre_day):
    down = []
    for i in range((size+pre_day),(len(seq)-size+1)):
        subset = seq[i:(i+size)]
        down.append(subset)
    print(type(down))
    return np.array(down)


def split_label(seq, size,pre_day):
    pre = []
    
    for i in range((size),(len(seq)-size-pre_day+1)):
        subset = seq[i:(i+pre_day)]
        pre.append(subset)

    print(type(pre))
    return np.array(pre)

def predict(train1, train2, size,pre_day):
    train1_pre_size = train1[len(train1)-size:]
    train2_pre_size = train2[:size]
    train1_pre_size = train1_pre_size.reshape((1,size,9))
    train2_pre_size = train2_pre_size.reshape((1,size,9))
    return train1_pre_size, train2_pre_size



from sklearn.model_selection import train_test_split

train1_up, train1_down, train1_label = make_set(train1_list)
train2_up, train2_down, train2_label = make_set(train2_list)
pred_up, pred_down = predict(train1_list, train2_list,train1_up.shape[1],5)
x_train_up = np.vstack([train1_up,train2_up])
x_train_down = np.vstack([train1_down,train2_down])
x_train_down = x_train_down.flatten()
re_x_train_down = []
for i in range(1,len(x_train_down)+1):
    re_x_train_down.append(x_train_down[-i])

re_x_train_down = np.array(re_x_train_down)
x_train_down = re_x_train_down

x_train_down = x_train_down.reshape((-1,train1_up.shape[1],train1_up.shape[2]))
    
x_train_label = np.vstack([train1_label,train2_label])

x_train_label = x_train_label[:,:,:-1]
x_train_label = x_train_label.reshape((-1,np.prod(x_train_label.shape[1:])))
# x_train_label = stan.inverse_transform(x_train_label)
# print("x_train_label",x_train_label[:5])

print(x_train_label.shape)
_,_,x_train_label,y_test_label = train_test_split(x_train_up,x_train_label, random_state=666, test_size = 0.2)
x_train_up,x_test_up,x_train_down,x_test_down = train_test_split(x_train_up,x_train_down, random_state=666, test_size = 0.2)
print(x_train_up.shape,x_train_down.shape,x_train_label.shape)
print("pred_set",pred_up.shape,pred_down.shape)








#### MODEL

from keras.models import *
from keras.layers import *
import tensorflow as tf

# leaky_relu = tf.nn.leaky_relu
input1 = Input(shape=(x_train_up.shape[1],x_train_up.shape[2]))
lstm1 = LSTM(5, activation=tf.nn.leaky_relu,return_sequences=True)(input1)
lstm1 = Dropout(0.7)(lstm1)
# lstm1 = BatchNormalization()(lstm1)
lstm1 = LSTM(20, activation=tf.nn.leaky_relu,return_sequences=True)(lstm1)
lstm1 = Dropout(0.4)(lstm1)
lstm1 = LSTM(10, activation="relu",return_sequences=True)(lstm1)

lstm1 = LSTM(10, activation=tf.nn.leaky_relu)(lstm1)
lstm1 = Dropout(0.3)(lstm1)


input2 = Input(shape=(x_train_up.shape[1],x_train_up.shape[2]))
lstm2 = LSTM(30, activation=tf.nn.leaky_relu,return_sequences=True)(input2)
lstm2 = Dropout(0.7)(lstm2)

# lstm2 = BatchNormalization()(lstm2)
lstm2 = LSTM(20, activation=tf.nn.leaky_relu,return_sequences=True)(lstm2)

lstm2 = LSTM(10, activation=tf.nn.leaky_relu)(lstm2)
lstm2 = Dropout(0.3)(lstm2)

merge1 = concatenate([lstm1, lstm2])
# middle1 = LSTM(100,activation="relu")(merge1)
middle1 = Dropout(0.3)(merge1)
middle1 = Dense(100,activation="relu")(middle1)
middle1 = Dense(80,activation="relu")(middle1)
middle1 = Dense(50,activation="relu")(middle1)
middle1 = Dropout(0.3)(middle1)


middle1 = Dense(40,activation="relu")(middle1)


model = Model(inputs = [input1, input2], outputs=middle1)
# model = Model(inputs = input1, outputs=lstm1)

model.summary()

model.compile(loss="MSE", optimizer="adadelta", metrics=['mae'])
model.fit([x_train_up,x_train_down],[x_train_label], epochs=100, batch_size=100)
# model.fit(x_train_up,x_train_label, epochs=50, batch_size=100)

y_=model.predict([x_test_up,x_test_down])
# y_ = stan.inverse_transform(y_)

# y_=model.predict([x_test_up])

y_ = np.round(y_)
y_[y_<0] = 0
y_[y_>9] = 9
print(y_)
y_ = np.array(y_).flatten()
y_test_label = y_test_label.flatten()
# y_test_label = stan.inverse_transform(y_test_label)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
def RMSE(y_test, y_):
    return np.sqrt(mean_squared_error(y_test,y_))
def RMAE(y_test, y_):
    return np.sqrt(mean_absolute_error(y_test,y_))

print("RMSE:",RMSE(y_test_label,y_))
print("RMAE:",RMAE(y_test_label,y_))
print("r2: ",r2_score(y_test_label,y_))






y_=model.predict([pred_up,pred_down])
# y_ = stan.inverse_transform(y_)
print(y_)
# y_=model.predict([pred_up])

y_=np.ceil(y_).astype(int)
y_[y_<0] = 0
y_[y_>9] = 9

y_ = np.array(y_).flatten()






y_ = y_.reshape((5,8)).astype(int)

np.savetxt("./kp_model/test0822_jyj.csv",y_,delimiter=",",fmt='%i')













    
    
    




