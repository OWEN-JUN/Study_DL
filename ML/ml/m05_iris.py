from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from keras.models import *
from keras.layers import *
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


seed =0
np.random.seed(seed)
tf.set_random_seed(seed)
def minmax_scaler(x,a):
    scaler =MinMaxScaler()
    scaler.fit(x[:,a:a+1])    
    x[:,a:a+1] = scaler.transform(x[:,a:a+1])
    return x

# dataset =np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
# x=dataset[:,:-1]
# y=dataset[:,-1]


# x = minmax_scaler(x,1)
# x = minmax_scaler(x,2)
# x = minmax_scaler(x,3)


def name_class(y):
    for i in range(len(y)):
        if y[i] == b"Iris-setosa":
            y[i] = 0
        elif y[i] == b"Iris-versicolor":
            y[i] = 1
        else:
            y[i] = 2

    return y
#Iris-setosa,Iris-versicolor,Iris-virginica "./data/iris.csv"
# dataset =np.loadtxt("./data/iris.csv", delimiter=",",dtype=None, names=('sepal length', 'sepal width', 'petal length', 'petal width', 'label'))
# dataset = np.loadtxt("./data/iris.csv",
#    dtype={'names': ('sepal length', 'sepal width', 'petal length', 'petal width', 'label'),
#           'formats': (np.float, np.float, np.float, np.float, '|S15')},
#    delimiter=',', skiprows=0)

x = np.loadtxt("./data/iris.csv", delimiter=",", usecols=[0,1,2,3])
y = np.loadtxt("./data/iris.csv", delimiter=",", usecols=[4],dtype='|S15')


from keras.utils import np_utils

# print(y)


y = name_class(y)
# print(y)
y = np.array(y,dtype=np.int32)
y = np_utils.to_categorical(y,3)
print(y)



import keras
model = Sequential()
model.add(Dense(60,input_dim=4, activation="relu"))
model.add(Dense(20, activation="relu"))
model.add(Dense(5, activation="relu"))

model.add(Dense(3, activation="softmax"))

model.compile(loss="binary_crossentropy", optimizer="adadelta", metrics=["acc"])

model.fit(x, y, epochs=3000, batch_size=30, validation_split=0.1)

print("Acc : ", model.evaluate(x,y)[1])
# print("predict: \n", model.predict_classes(x))
