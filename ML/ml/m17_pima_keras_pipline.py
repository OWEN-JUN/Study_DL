
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
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import seaborn as sns
import keras
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
pima = pd.read_csv("./data/pima-indians-diabetes.csv",header = None)

# print(pima.head())

x = pima.iloc[:,:-1]
y = pima.iloc[:,-1]
print(x.shape)
print(y.shape)
x, x_test, y, y_test = train_test_split(x,y,test_size=0.15)
def build_network_cnn(keep_prob=0.5, optimizer="adam"):
    inputs = Input(shape=(8,), name="input")
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
parameters_knn ={"knn__n_neighbors":[1,2,3,4,5],"knn__weights":["uniform","distance"],"knn__leaf_size":[10,20,30],"knn__algorithm":["ball_tree","brute","kd_tree"]}
parameters_svc =  {"svm__C":[1,10,100,1000],"svm__kernel":["linear","rbf","sigmoid"],"svm__gamma":[0.001,0.0001]}


parameters_rf = {"rf__n_estimators":[10,20,30,40],"rf__max_features":[1,2,3],"rf__max_depth":[10,20,30]}
    
                
def create_hyperparameters():
    batches=[5,7,9,10]
    optimizers = ["SGD", "adam"]
    dropout = np.linspace(0,0.1, 2)
    return{"model__batch_size":batches, "model__optimizer":optimizers,"model__keep_prob":dropout}
               
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras.wrappers.scikit_learn import KerasClassifier
model =KerasClassifier(build_fn = build_network_cnn, verbose=1)


pip = Pipeline([("scaler", MinMaxScaler()),("model",model)])
pip2 = Pipeline([("scaler", MinMaxScaler()),("svm",SVC())])
pip3 = Pipeline([("scaler", MinMaxScaler()),("knn",KNeighborsClassifier())])
pip4 = Pipeline([("scaler", MinMaxScaler()),("rf",RandomForestClassifier())])
# svc,kneighbors,random

hyper = create_hyperparameters()
clf4 = RandomizedSearchCV(estimator=pip, param_distributions=hyper, n_iter=10,n_jobs=5, cv=3, verbose=1)
clf5 = RandomizedSearchCV(estimator=pip2,param_distributions=parameters_svc,n_iter=10, n_jobs=5, cv=4, verbose=1)
clf6 = RandomizedSearchCV(estimator=pip3,param_distributions=parameters_knn,n_iter=18, n_jobs=5, cv=5, verbose=1)
clf7 = RandomizedSearchCV(estimator=pip4,param_distributions=parameters_rf,n_iter=9, n_jobs=5, cv=5, verbose=1)



clf4.fit(x,y)
clf5.fit(x,y)
clf6.fit(x,y)
clf7.fit(x,y)
y_pred = clf4.predict(x)
y_pred = clf5.predict(x)
y_pred = clf6.predict(x)

print('knn@@@@@@@@@@@@@@@@@@@randomsearch')
print("최적의 매개 변수 : ", clf4.best_estimator_)
print("최적의 매개 변수 : ", clf5.best_estimator_)

y_pred = clf4.predict(x)
print("최종 정답률", accuracy_score(y, y_pred))

last_score = clf4.score(x,y)
print("최종 정답률", last_score)

clf4.best_estimator_


model_dic =clf4.best_params_
from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor="loss", patience=50, mode="auto")
model = build_network_cnn(model_dic["model__keep_prob"],model_dic["model__optimizer"])
model.fit(x, y,batch_size=model_dic["model__batch_size"], epochs=500, callbacks=[early], validation_split=0.2)

print("acc:", model.evaluate(x_test,y_test))




# model_dic = clf5.best_params_

# model2.fit(x,y)

# print(model2.score(x_test,y_test))

model_dic = clf6.best_params_
model2 = KNeighborsClassifier(n_neighbors=model_dic["knn__n_neighbors"],weights=model_dic["knn__weights"],leaf_size=model_dic["knn__leaf_size"],algorithm=model_dic["knn__algorithm"])
model2.fit(x,y)

print(model2.score(x_test,y_test))




model_dic = clf7.best_params_
model3 = RandomForestClassifier(n_estimators=model_dic["rf__n_estimators"],max_features=model_dic["rf__max_features"],max_depth=model_dic["rf__max_depth"])
model3.fit(x,y)

print(model3.score(x_test,y_test))