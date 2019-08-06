from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./data/tem10y.csv", encoding="utf-8")
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
lr = RandomForestRegressor(max_depth=30, n_estimators=1000)
lr.fit(train_x, train_y)
pre_y = lr.predict(test_x)

plt.figure(figsize=(10,6), dpi=100)
plt.plot(test_y, c="r")
plt.plot(pre_y, c="b")
plt.savefig('tenki-kion-lr.png')
plt.show()
from sklearn.metrics import r2_score
r2 = r2_score(test_y, pre_y)
print(lr.score(test_x, test_y))