import pandas as pd

import numpy as np
from sklearn.preprocessing import StandardScaler
test = pd.read_csv("./data/test0822.csv")
test_columns = test.columns.drop("date")
print(test_columns)
print(test)


max = test.max() ##max = 9
min = test.min() ##min = 0

print(max,min)

test["kp_sum"] =test[test_columns].sum(axis=1)
test["k_wind"] = 0


print(test)
print(len(test))

train1 = test[:3113]
train2 = test[3118:]
l1 = test[3113:3118]

stan = StandardScaler()

train1_sum_list = np.array(train1["kp_sum"]).reshape((-1,1))
train2_sum_list = np.array(train2["kp_sum"]).reshape((-1,1))
stan.fit(train1_sum_list)
train1["k_wind"] = stan.transform(train1_sum_list)
train2["k_wind"] = stan.transform(train2_sum_list)
train1 = train1.drop(["date","kp_sum"],axis=1)
train2 = train2.drop(["date","kp_sum"],axis=1)
train1_list = np.array(train1)
train2_list = np.array(train2)
print(len(train1))
print(len(train2))
print(train1_list.shape)
print(train2.shape)
print(l1)


###예측에 쓰이는 기간


def predict(train1, train2, size,pre_day):
    train1_pre_size = train1[len(train1)-size:]
    train2_pre_size = train2[:size]
    return train1_pre_size, train2_pre_size



from sklearn.model_selection import train_test_split


pred_up, pred_down = predict(train1_list,train2_list,10,5)

from keras.models import load_model
from keras.models import *
from keras.layers import *
import tensorflow as tf

leaky_relu = tf.nn.leaky_relu
model = load_model('./kp_model/kp_v1.h5')


day5_p = model.predict([pred_up,pred_down])
print(day5_p)