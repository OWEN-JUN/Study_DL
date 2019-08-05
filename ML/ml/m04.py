from sklearn.svm import LinearSVC, SVC

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from keras.models import *
from keras.layers import *
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import keras

seed =0
np.random.seed(seed)
tf.set_random_seed(seed)
def minmax_scaler(x,a):
    scaler =MinMaxScaler()
    scaler.fit(x[:,a:a+1])    
    x[:,a:a+1] = scaler.transform(x[:,a:a+1])
    return x

dataset =np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
x=dataset[:,:-1]
y=dataset[:,-1]


# x = minmax_scaler(x,1)
# x = minmax_scaler(x,2)
# x = minmax_scaler(x,3)

#모델
# clf = KNeighborsClassifier(n_neighbors=1,)
clf = SVC()
#실행
clf.fit(x, y)

y_ =clf.predict(x)
print("Acc : ", accuracy_score(y,y_)) #accuracy_score는 분류에서만
# print("predict: \n")
