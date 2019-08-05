from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
#데이터
learn_data = [[0,0],[1,0],[0,1],[1,1]]
learn_label = [0,0,0,1]
#모델
clf = LinearSVC()
#실행
clf.fit(learn_data, learn_label)


#평가예측

test_data = [[0,0],[1,0],[0,1],[1,1]]
y_pre = clf.predict(test_data)
print(y_pre) 


print("acc: ", accuracy_score([0,0,0,1],y_pre))


