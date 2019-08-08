from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import mglearn
cancer = load_breast_cancer()
x= cancer.data

# x = np.delete(x,[10,11,12,13,14,15,20,21,25,1,2,18,19,0],1)
x_train, x_test, y_train, y_test = train_test_split(x, cancer.target,stratify=cancer.target, random_state=42)

# tree = DecisionTreeClassifier(random_state=0)
# tree.fit(x_train, y_train)
# print("훈련 세트 정확도 :",tree.score(x_train,y_train))
# print("테스트 세트 정확도 :",tree.score(x_test,y_test))


tree = XGBClassifier(max_depth=8, random_state=0, n_estimators=300)

tree.fit(x_train, y_train)
print("훈련 세트 정확도 :",tree.score(x_train,y_train))
print("테스트 세트 정확도 :",tree.score(x_test,y_test))


print("특성 중요도 : ",len(tree.feature_importances_))
print("특성 중요도 : ",tree.feature_importances_)


def plot_feature_importances_cancer(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), tree.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("특성중요도")
    plt.ylabel("특성")
    plt.ylim(-1,n_features)

plot_feature_importances_cancer(tree)
plt.show()


####결과
# 훈련 세트 정확도 : 1.0
# 테스트 세트 정확도 : 0.965034965034965