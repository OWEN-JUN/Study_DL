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
import pandas as pd
seed =0
np.random.seed(seed)
tf.set_random_seed(seed)
def minmax_scaler(x,a):
    scaler =MinMaxScaler()
    scaler.fit(x[:,a:a+1])    
    x[:,a:a+1] = scaler.transform(x[:,a:a+1])
    return x
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
iris_data = pd.read_csv("./data/iris.csv", encoding="utf-8", names = ["SepalLength", "SepalWidth","PetalLength", "PetalWidth","Name"])


print(iris_data)
print(iris_data.shape)


#붓꽃 데이터를 레이블과 입력 데이터로 분리하기
x = iris_data.loc[:,["SepalLength", "SepalWidth","PetalLength", "PetalWidth"]]
y = iris_data.loc[:,"Name"]

# x1 = iris_data.iloc[:,4]
# x2 = iris_data.iloc[:,0:4]



# x = np.loadtxt("./data/iris.csv", delimiter=",", usecols=[0,1,2,3])
# y = np.loadtxt("./data/iris.csv", delimiter=",", usecols=[4],dtype='|S15')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, train_size=0.8, shuffle=True)

from keras.utils import np_utils

# print(y)

#케라스 원핫
# y = name_class(y)
# print(y)
# y = np.array(y,dtype=np.int32)
# y = np_utils.to_categorical(y,3)
# print(y)
# print(y)


#넘파이 원핫
# y = y.reshape((-1,1))
# from sklearn.preprocessing import OneHotEncoder
# oneh = OneHotEncoder()
# oneh.fit(y)
# y = oneh.transform(y).toarray()
# print(y) 


#모델
# clf = KNeighborsClassifier(n_neighbors=1,)
# clf = MLPClassifier(activation="relu",batch_size=15,validation_fraction=0.8, verbose=True,solver='adam', max_iter=3000,n_iter_no_change=1000 )

# clf =SVC()
# clf = KNeighborsClassifier(n_neighbors=1)
clf = LinearSVC()
#실행
clf.fit(x_train, y_train)

y_ =clf.predict(x_test)
print("Acc : ", accuracy_score(y_test,y_)) #accuracy_score는 분류에서만
# print("predict: \n")



#머신러닝에서는 원핫이 아닌 스트링으로도 된다