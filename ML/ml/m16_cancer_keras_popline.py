
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
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import seaborn as sns
import keras
import matplotlib.pyplot as plt
cancer = load_breast_cancer()

# print(cancer)
print(cancer.DESCR)
x = cancer.data
y = cancer.target
# print(x.shape)
# print(y)
# plt.figure(figsize=(15,15))
# sns.pairplot(np.array(x), hue=y)
# plt.show()

# import sys
# sys.exit()


def build_network_cnn(keep_prob=0.5, optimizer="adam"):
    inputs = Input(shape=(30,), name="input")
    x = Dense(60, activation="relu",name="d1")(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(100, activation="relu",name="d2")(x)
    x = Dense(50, activation="relu",name="d3")(x)
    x = Dense(30, activation="relu",name="d4")(x)
    x = Dropout(keep_prob)(x)
    x = Dense(10, activation="relu",name="d5")(x)
 
    prediction = Dense(1, activation="sigmoid",name='output')(x)
    model = Model(input = inputs, outputs=prediction)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["acc"])
    
    return model


def create_hyperparameters():
    batches=[5,7,9,10]
    optimizers = ["SGD", "adam"]
    dropout = np.linspace(0,0.1, 2)
    return{"model__batch_size":batches, "model__optimizer":optimizers,"model__keep_prob":dropout}



# parameters_svc =  {"svm__C":[1,10,100,1000],"svm__kernel":["linear","rbf","sigmoid"],"svm__gamma":[0.001,0.0001]}
parameters_svc =  {"C":[1,10,100,1000],"kernel":["linear","rbf","sigmoid"],"gamma":[0.001,0.0001]}

# hyperparameters = create_hyperparameters()



from keras.wrappers.scikit_learn import KerasClassifier
model =KerasClassifier(build_fn = build_network_cnn, verbose=1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
# pip = Pipeline([("scaler", MinMaxScaler()),("svm",SVC())])



# search=RandomizedSearchCV(estimator=pip, param_distributions=parameters_svc, n_iter=10,n_jobs=5, cv=3, verbose=1)
search=RandomizedSearchCV(estimator=SVC(), param_distributions=parameters_svc, n_iter=10,n_jobs=5, cv=3, verbose=1)
pip = make_pipeline(MinMaxScaler(),search)

# print(y_train.shape)
pip.fit(x, y)

# # 
print(search.best_params_)
print("best score : ", search.best_score_)
print("score:",search.score(x,y))

model_dic =search.best_params_
# from keras.callbacks import EarlyStopping
# early = EarlyStopping(monitor="loss", patience=50, mode="auto")
# model = build_network_cnn(model_dic["svm__C"],model_dic["model__optimizer"])
# model.fit(x, y,batch_size=model_dic["model__batch_size"], epochs=500, callbacks=[early])

# print("acc:", model.evaluate(x,y))