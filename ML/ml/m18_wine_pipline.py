import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding="utf-8")
import seaborn as sns
# plt.figure(figsize=(15,15))
# sns.pairplot(wine)
# plt.show()
y = wine["quality"]
# x = wine.drop(["quality"], axis=1)
x = wine.drop(["fixed acidity","volatile acidity","residual sugar",
                "chlorides","density","quality"], axis=1)

from keras.utils import np_utils
newlist=[]
for v in list(y):
    if v<=4:
        newlist += [0]
    elif v<=7:
        newlist += [1]
    else:
        newlist += [2]

y = newlist
y = np_utils.to_categorical(y)
print(y)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import keras
def build_network_cnn(keep_prob=0.5, optimizer="adam",node1=10,node2=10,node3=10,layer_cnt=0):
    inputs = Input(shape=(6,), name="input")
    x = Dense(node1, activation="relu",name="d1")(inputs)
    x = Dropout(keep_prob)(x)
    for i in range(layer_cnt):
        x = Dense(node2, activation="relu")(x)

    x = Dense(node2, activation="relu",name="d2")(x)
    x = Dense(node3, activation="relu",name="d3")(x)
    x = Dense(10, activation="relu",name="d4")(x)
 
    prediction = Dense(3, activation="softmax",name='output')(x)
    model = Model(input = inputs, outputs=prediction)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])
    
    return model
def create_hyperparameters():
    batches=[5,7,9,10]
    optimizers = ["SGD", "adam"]
    dropout = np.linspace(0,0.1, 2)
    node1 = [30,40,50,60]
    node2 = [30,40,50,60]
    node3 = [30,40,50,60]
    layer_cnt = [2,3,4,5]
    return{"model__batch_size":batches, 
            "model__optimizer":optimizers,
            "model__keep_prob":dropout,
            "model__node1" :node1,
            "model__node2" :node2,
            "model__node3" :node3,
            "model__layer_cnt":layer_cnt}


hyperparameters = create_hyperparameters()

model =KerasClassifier(build_fn = build_network_cnn, verbose=1)


pip = Pipeline([("scaler", MinMaxScaler()),("model",model)])
model=RandomizedSearchCV(estimator=pip, param_distributions=hyperparameters, n_iter=10,n_jobs=5, cv=3, verbose=1)



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)

# model = RandomForestClassifier(n_estimators=1000,max_depth=1000,max_leaf_nodes=1000,oob_score=True,)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)
# print(classification_report(y_test, y_pred))
print("score : ",model.score(x_test, y_test))
# print("score : ",model.score(x_train, y_train))
print(y_pred)
# print("정답률", accuracy_score(y_test, y_pred))

print(model.best_params_)
model_dic =model.best_params_
from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor="loss", patience=50, mode="auto")
model = build_network_cnn(model_dic["model__keep_prob"],model_dic["model__optimizer"],model_dic["model__node1"],
                            model_dic["model__node2"],
                            model_dic["model__node3"],
                            model_dic["model__layer_cnt"])
model.fit(x, y,batch_size=model_dic["model__batch_size"], epochs=500, callbacks=[early], validation_split=0.2)

print("acc:", model.evaluate(x_test,y_test))