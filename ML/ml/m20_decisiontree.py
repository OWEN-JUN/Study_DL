from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target,stratify=cancer.target, random_state=42)

# tree = DecisionTreeClassifier(random_state=0)
# tree.fit(x_train, y_train)
# print("훈련 세트 정확도 :",tree.score(x_train,y_train))
# print("테스트 세트 정확도 :",tree.score(x_test,y_test))


tree = DecisionTreeClassifier(max_depth=1000, random_state=0)
tree.fit(x_train, y_train)
print("훈련 세트 정확도 :",tree.score(x_train,y_train))
print("테스트 세트 정확도 :",tree.score(x_test,y_test))


print("특성 중요도 : ",len(tree.feature_importances_))
print("특성 중요도 : ",tree.feature_importances_)