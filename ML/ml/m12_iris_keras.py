
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from keras.models import *
from keras.layers import *
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
iris_data = pd.read_csv("./data/iris2.csv", encoding="utf-8")

# print(y)

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)
print(y_test)

# y = name_class(y)
# print(y)
# y = np.array(y,dtype=np.int32)
# y = np_utils.to_categorical(y,3)

from sklearn.model_selection import RandomizedSearchCV
import keras
def build_network_cnn(keep_prob=0.5, optimizer="adam"):
    inputs = Input(shape=(4,), name="input")
    x = Dense(60, activation="relu",name="d1")(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(60, activation="relu",name="d2")(x)
    x = Dense(60, activation="relu",name="d3")(x)
 
    prediction = Dense(3, activation="softmax",name='output')(x)
    model = Model(input = inputs, outputs=prediction)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])
    
    return model

# def build_network(keep_prob=0.5, optimizer='adam'):
#     inputs = Input(shape=(4, ), name='input')
#     x = Dense(1024, activation='relu', name='hidden1')(inputs)
#     x = Dropout(keep_prob)(x)
#     x = Dense(512, activation='relu', name='hidden2')(x)
#     x = Dropout(keep_prob)(x)
#     x = Dense(256, activation='relu', name='hidden3')(x)
#     x = Dropout(keep_prob)(x)
#     x = Dense(128, activation='relu', name='hidden4')(x)
#     x = Dropout(keep_prob)(x)
#     x = Dense(64, activation='relu', name='hidden5')(x)
#     x = Dropout(keep_prob)(x)
#     prediction = Dense(3, activation='softmax', name='output')(x)
#     model = Model(inputs=inputs, output=prediction)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#     return model


def create_hyperparameters():
    batches=[10,20,30,40,50]
    optimizers = ["SGD", "adam", "adadelta"]
    dropout = np.linspace(0.1,0.5, 5)
    return{"batch_size":batches, "optimizer":optimizers,"keep_prob":dropout}
from keras.wrappers.scikit_learn import KerasClassifier
hyperparameters = create_hyperparameters()

model =KerasClassifier(build_fn = build_network_cnn, verbose=1)
search=RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=10,n_jobs=1, cv=3, verbose=1)
# print(y_train.shape)
search.fit(x_train, y_train)

# # 
print(search.best_params_)
print("score : ", search.best_score_)

##결과값 {'optimizer': 'SGD', 'keep_prob': 0.30000000000000004, 'batch_size': 50}

print(x_train.shape)
print(y_train.shape)