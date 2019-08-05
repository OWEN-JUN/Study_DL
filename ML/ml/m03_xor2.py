from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#데이터
learn_data = [[0,0],[1,0],[0,1],[1,1]]
learn_label = [0,1,1,0]
#모델
clf = KNeighborsClassifier(n_neighbors=1)
#실행
clf.fit(learn_data, learn_label)


#평가예측

test_data = [[0,0],[1,0],[0,1],[1,1]]
y_pre = clf.predict(test_data)
print(y_pre) 


print("acc: ", accuracy_score([0,1,1,0],y_pre))


