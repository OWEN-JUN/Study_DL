import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding="utf-8")
import seaborn as sns
# plt.figure(figsize=(15,15))
# sns.pairplot(wine)
# plt.show()
y = wine["quality"]
# x = wine.drop(["quality"], axis=1)
x = wine.drop(["fixed acidity","volatile acidity","residual sugar",
                "chlorides","density","quality"], axis=1)


newlist=[]
for v in list(y):
    if v<=4:
        newlist += [0]
    elif v<=7:
        newlist += [1]
    else:
        newlist += [2]

y = newlist



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)

model = RandomForestClassifier(n_estimators=1000,max_depth=1000,max_leaf_nodes=1000,oob_score=True,)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("score : ",model.score(x_test, y_test))
# print("score : ",model.score(x_train, y_train))
print(y_pred)
print("정답률", accuracy_score(y_test, y_pred))