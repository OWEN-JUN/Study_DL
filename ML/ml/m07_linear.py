from sklearn.datasets import load_boston
import numpy as np
boston = load_boston()
# boston_tar = load_boston().target()
print(boston.data.shape)
print(boston.keys())
print(boston.target)
print(boston.target.shape)

x = boston.data
y = boston.target

print(type(boston))

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


model1 = LinearRegression() 
model2 = Ridge()
model3 = Lasso()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)



print("LinearRegression: ", model1.score(x_test,y_test))
print("Ridge: ",model2.score(x_test,y_test))
print("Lasso: ",model3.score(x_test,y_test))