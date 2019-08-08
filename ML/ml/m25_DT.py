import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *

wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding="utf-8")

# plt.figure(figsize=(15,15))
# sns.pairplot(wine)
# plt.show()
y = wine["quality"]
# x = wine.drop(["quality"], axis=1)
x = wine.drop(["fixed acidity","volatile acidity","residual sugar",
                "chlorides","density","quality"], axis=1)
label = x.columns
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

# print(y)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import keras


parameters_dt = {"max_depth":[10,20,30]}
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)
kfold_cv = KFold(n_splits=5, shuffle=True)

model=RandomizedSearchCV(estimator=DecisionTreeClassifier(), param_distributions=parameters_dt, n_iter=30,n_jobs=5, cv=kfold_cv, verbose=1)




model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)
# print(classification_report(y_test, y_pred))
print("score : ",model.score(x_test, y_test))
# print("score : ",model.score(x_train, y_train))
print(y_pred)
print("정답률", accuracy_score(y_test, y_pred))

# print("특성 중요도 : ",len(model.feature_importances_))
# print("특성 중요도 : ",model.feature_importances_)


# def plot_feature_importances_cancer(model):
#     n_features = x_test.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align="center")
#     plt.yticks(np.arange(n_features), label)
#     plt.xlabel("특성중요도")
#     plt.ylabel("특성")
#     plt.ylim(-1,n_features)

# plot_feature_importances_cancer(model)
# plt.show()


#결과
# 0.9163265306122449