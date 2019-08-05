from keras.layers import *
from keras.models import *





learn_data = np.array([[0,0],[1,0],[0,1],[1,1]])
learn_label = np.array([0,0,0,1])
#모델

model = Sequential()
# clf = LinearSVC()


model.add(Dense(1, input_shape=(2,)))
# model.add(Dense(1,))
# model.add(Activation("sigmoid"))
# model.add(Dense(1,activation="sigmoid"))

#실행
# clf.fit(learn_data, learn_label)
model.compile(loss="mse",optimizer="adadelta",metrics=["acc"])
model.fit(learn_data, learn_label, batch_size=1, epochs=2000)


#평가예측

test_data = np.array([[0,0],[1,0],[0,1],[1,1]])
test_label = np.array([0,0,0,1])

# y_pre = clf.predict(test_data)
y_pre =np.abs(np.round(model.predict(test_data))) 
print(y_pre) 


# print("acc: ", accuracy_score([0,0,0,1],y_pre))
_, acc = model.evaluate(test_data, test_label)

print("acc: ", acc)

