from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import keras
from xgboost import XGBRegressor
df = pd.read_csv("../data/tem10y.csv", encoding="utf-8")
train_year = (df["연"] <=2015)
test_year = (df["연"]>=2016)
interval=6

def make_data(data):
    x=[]
    y=[]
    temps=list(data["기온"])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa=[]
        for p in range(interval):
            d = i+p-interval
            xa.append(temps[d])
        x.append(xa)
    return (x,y)

train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])

# lr = LinearRegression(normalize=True)
# print(train_x)
# print(type(train_y))



parameters_dt = {"max_depth":[4,5,10,20,30,40],"n_estimators":[100,200,300,400,500,600],"gamma":[0.01,0.001,0.03],"min_child_weight":[3,4,5]}

kfold_cv = KFold(n_splits=5, shuffle=True)

model=RandomizedSearchCV(estimator=XGBRegressor(), param_distributions=parameters_dt, n_iter=50,n_jobs=5, cv=kfold_cv, verbose=1)


model.fit(train_x, train_y)
pre_y = model.predict(test_x)

# plt.figure(figsize=(10,6), dpi=100)
# plt.plot(test_y, c="r")
# plt.plot(pre_y, c="b")
# plt.savefig('tenki-kion-lr.png')
# plt.show()
from sklearn.metrics import r2_score
r2 = r2_score(test_y, pre_y)
print(model.score(test_x, test_y))
print(model.best_estimator_)



###################
# 0.9189123870451037
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0.001,
#              importance_type='gain', learning_rate=0.1, max_delta_step=0,
#              max_depth=4, min_child_weight=1, missing=None, n_estimators=100,
#              n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#              silent=None, subsample=1, verbosity=1)