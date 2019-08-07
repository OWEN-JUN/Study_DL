import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV,KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
warnings.filterwarnings("ignore")

iris_data = pd.read_csv("./data/iris2.csv", encoding="utf-8")

y = iris_data.loc[:,"Name"]
x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
warnings.filterwarnings("ignore")
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, train_size=0.8,shuffle=True)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

pip = Pipeline([("scaler", MinMaxScaler()),("svm",SVC())])
# from sklearn.pipeline import make_pipeline
# pip = make_pipeline(MinMaxScaler(),SVC())
pip.fit(x_train, y_train)

print("점수", pip.score(x_test,y_test))

