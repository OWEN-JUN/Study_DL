from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()
x= cancer.data
# x = np.delete(x,[0,1,2,4,6,7,8,11,14,15,16,17,18,19,22],1)
x_train, x_test, y_train, y_test = train_test_split(x, cancer.target,stratify=cancer.target, random_state=42)

# tree = DecisionTreeClassifier(random_state=0)
# tree.fit(x_train, y_train)
# print("훈련 세트 정확도 :",tree.score(x_train,y_train))
# print("테스트 세트 정확도 :",tree.score(x_test,y_test))


tree = DecisionTreeClassifier(max_depth=5, random_state=0)

tree.fit(x_train, y_train)
print("훈련 세트 정확도 :",tree.score(x_train,y_train))
print("테스트 세트 정확도 :",tree.score(x_test,y_test))


print("특성 중요도 : ",len(tree.feature_importances_))
print("특성 중요도 : ",tree.feature_importances_)


# def plot_feature_importances_cancer(model):
#     n_features = cancer.data.shape[1]
#     plt.barh(np.arange(n_features), tree.feature_importances_, align="center")
#     plt.yticks(np.arange(n_features), cancer.feature_names)
#     plt.xlabel("특성중요도")
#     plt.ylabel("특성")
#     plt.ylim(-1,n_features)

# plot_feature_importances_cancer(tree)
# plt.show()