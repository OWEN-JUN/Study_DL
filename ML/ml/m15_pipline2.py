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
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, train_size=0.8,shuffle=True)


parameters_svc =  {"svm__C":[1,10,100,1000],"svm__kernel":["linear","rbf","sigmoid"],"svm__gamma":[0.001,0.0001]}
             

parameters_rf = [
                {"n_estimators":[10,20,30,40],"max_features":[1,2,3]},
                {"n_estimators":[10,20,30,40],"max_features":[1,2,3],"max_depth":[10,20,30]}
                ]

parameters_knn ={"n_neighbors":[1,2,3,4,5],"weights":["uniform","distance"],"leaf_size":[10,20,30],"algorithm":["ball_tree","brute","kd_tree"]}
                # {"n_neighbors":[1,2,3,4,5],"leaf_size":[10,20,30],"algorithm":["kd_tree"]},
                # {"n_neighbors":[1,2,3,4,5],"leaf_size":[10,20,30],"algorithm":["brute"]},
                
                



def create_hyperparameters():
    n_neighbors=[1,2,3,4,5]
    weights = ["uniform","distance"]
    leaf_size = ["uniform","distance"]
    algorithm=["ball_tree","brute","kd_tree"]
    return{"n_neighbors":n_neighbors, "weights":weights,"leaf_size":leaf_size,"algorithm":algorithm}

                
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pip = Pipeline([("scaler", MinMaxScaler()),("svm",SVC())])
# svc,kneighbors,random
kfold_cv = KFold(n_splits=10, shuffle=True)
clf = GridSearchCV(SVC(),parameters_svc, cv=kfold_cv)
clf2 = GridSearchCV(RandomForestClassifier(),parameters_rf, cv=kfold_cv)
clf3= GridSearchCV(KNeighborsClassifier() ,parameters_knn, cv=kfold_cv)
model = KNeighborsClassifier()


clf4 = RandomizedSearchCV(estimator=pip,param_distributions=parameters_svc,n_iter=10, n_jobs=1, cv=3, verbose=1)



clf4.fit(x,y)


print('knn@@@@@@@@@@@@@@@@@@@randomsearch')
# print("최적의 매개 변수 : ", clf4.best_estimator_)

y_pred = clf4.predict(x)
print("최종 정답률", accuracy_score(y, y_pred))

last_score = clf4.score(x,y)
print("최종 정답률", last_score)
