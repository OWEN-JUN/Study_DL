import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
# warnings.filterwarnings("ignore")

# iris_data = pd.read_csv("./data/iris2.csv", encoding="utf-8")

# y = iris_data.loc[:,"Name"]
# x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
# warnings.filterwarnings("ignore")
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, train_size=0.8,shuffle=True)

# warnings.filterwarnings("ignore")

# allAlgorithms = all_estimators(type_filter="classifier")

# for(name, algorithm) in allAlgorithms:
#     clf = algorithm()
#     clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
#     print(name, "의 정답률:", accuracy_score(y_test, y_pred))



warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris2.csv", encoding="utf-8")

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = iris_data.loc[:, 'Name']

# from keras.utils import np_utils


# label = LabelEncoder()
# label.fit(y)
# y = label.transform(y)

# y = np_utils.to_categorical(y,3)



print(y)
# 학습 전용과 테스트 전용 분리하기


# classifier 알고리즘 모두 추출하기--- (*1)
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")
maxsplitelist=[]
n_splits_num = 3
while n_splits_num < 11:
    print("@@@@@@@@@@@@@@@@@@@@n_splits_num ===== %5d  @@@@@@@@@@@@@@@@@@@@"%n_splits_num)
    kfold_cv = KFold(n_splits=n_splits_num, shuffle=True)
    maxlist=[]
    for(name, algorithm) in allAlgorithms:
        # 각 알고리즘 객체 생성하기--- (*2)
        # print(name)
        clf = algorithm()
        # print(str(type(clf)) in "base_estimator")
        
        if hasattr(clf,"score"):

            scores = cross_val_score(clf,x,y,cv=kfold_cv)
            print(name,"의 정답률", scores)
            
            maxlist.append([name, np.mean(scores)])


    maxlist=sorted(maxlist, key=lambda x: x[:][1],reverse=True)  
    maxsplitelist.append(["n_splits:%d"%(n_splits_num),maxlist[:3]])
    n_splits_num += 1 
print(maxsplitelist)
    