import numpy as np
a = np.arange(10)
print(a)
np.save("aaa.npy",a)
b = np.load("aaa.npy")
print(b)

# ######모델저장####
# model.save("savetest01.h5")
# #####모델불러오기#####
# from keras.models import load_model
# model = load_model("savetest01.h5")
# from keras.layers import Dense
# model.add(Dense(1))

########pandas to numpy#############3
# pandas.value

########## csv 불러오기############
# dataset = numpy.loadtxt("000.csv",delimiter=",")
# iris_data = pd.read_csv("000.csv",encoding="utf-8")
        # index_col = 0, encoding="cp949", sep=",", header=none
        #names=["x1","x2"]
# wine = pd.read_csv("000.csv", sep=",",encoding="utf-8")

####utf-8#####
# -*-coding: utf-8 -*-
#####sample######
def name_class(y):
    for i in range(len(y)):
        if y[i] == b"Iris-setosa":
            y[i] = 0
        elif y[i] == b"Iris-versicolor":
            y[i] = 1
        else:
            y[i] = 2

    return y
import pandas as pd
from keras.datasets import mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()



mnist_x = np.vstack((np.array(x_train), np.array(x_test)))
mnist_y = np.hstack((np.array(y_train), np.array(y_test)))
np.save("mnist_x.npy",mnist_x)
np.save("mnist_y.npy",mnist_y)
x = np.load("mnist_x.npy")
y = np.load("mnist_y.npy")

# mnist_train =
print(x.shape)
print(y.shape)




from keras.datasets import cifar10
(x_train, y_train),(x_test,y_test) = cifar10.load_data()
cifar10_x = np.vstack((np.array(x_train), np.array(x_test)))
cifar10_y = np.vstack((np.array(y_train), np.array(y_test)))
np.save("cifar10_x.npy",cifar10_x)
np.save("cifar10_y.npy",cifar10_y)
x = np.load("cifar10_x.npy")
y = np.load("cifar10_y.npy")
# mnist_train =
print(x.shape)
print(y.shape)

from keras.datasets import boston_housing
(x_train, y_train),(x_test,y_test) = boston_housing.load_data()
boston_housing_x = np.vstack((np.array(x_train), np.array(x_test)))
boston_housing_y = np.hstack((np.array(y_train), np.array(y_test)))

np.save("boston_housing_x.npy",boston_housing_x)
np.save("boston_housing_y.npy",boston_housing_y)
x = np.load("boston_housing_x.npy")
y = np.load("boston_housing_y.npy")
# mnist_train =
print("boston_housingx",x.shape)
print("boston_housingy",y.shape)




# from skleran.datasets import load_boston

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
label = cancer.target.reshape(-1,1)
print("cancer_ori",cancer.data.shape)
print("cancer_ori",cancer.target.shape)
cancer_data = np.c_[cancer.data,label]

np.save("cancer_data.npy",cancer_data)
cancer_d = np.load("cancer_data.npy")
print("cancer",cancer_d.shape)





iris_data = pd.read_csv("../data/iris2.csv", encoding="utf-8")


x = np.array(iris_data.iloc[:,:-1])
y = name_class(iris_data.iloc[:,-1])

y = np.array(y,dtype=np.int32)
iris2_data = np.c_[x,y]
np.save("iris2_data.npy",iris2_data)
# np.save("iris2_label.npy",y)

iris2_data = np.load("./iris2_data.npy")
# iris2_label = np.load("./iris2_label.npy")

print("iris2_data:",iris2_data.shape)
# print("iris2_label:",iris2_label.shape)

wine_data = pd.read_csv("../data/winequality-white.csv",sep=";", encoding="utf-8")
# print(wine_data)
np.save("wine_data.npy",np.array(wine_data))
wine = np.load("wine_data.npy")
print("whine:",wine.shape)
pima = pd.read_csv("../data/pima-indians-diabetes.csv",header = None)
# print(pima)
np.save("pima.npy",np.array(pima))
pima = np.load("pima.npy")
print("pima:",pima.shape)



