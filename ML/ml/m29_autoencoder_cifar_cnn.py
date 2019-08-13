from keras.datasets import mnist
from keras.datasets import cifar10
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# (x_train, _),(x_test, _) = mnist.load_data()
(x_train, _),(x_test, _) = cifar10.load_data()

# print(x_train[0])
minma = MinMaxScaler()
x_train = x_train.reshape((len(x_train),x_train.shape[1],x_train.shape[2],3))

x_train = np.array(x_train, dtype=np.float32)
# print(x_train.shape)
# print(x_train[0])



x_test = x_test.reshape((len(x_test),x_test.shape[1],x_test.shape[2],3))
print(x_train.shape)
print(x_test.shape)



###모델구성###

from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt

def build_network_dnn(optimizer = "adam",keep_prob=0.5,encoding_dim = 30):
    encoding_dim = encoding_dim
    input_img = Input(shape=(32,32,3))

    x = Conv2D(32,(3,3),padding="same")(input_img)
    x = Conv2D(32,(3,3),padding="same")(x)
    
    encoded = Conv2D(32,(3,3),padding="same")(x)
    # x = Dense(3072, activation="relu")(encoded)
    x = Conv2D(32,(3,3),padding="same")(encoded)
    #디코더는 입력의 손실있는 재구성 (lossy reconstryction)
    decoded = Conv2D(3,(3,3),padding="same")(x)
    # decoded = Dense(784, activation="relu")(encoded)





    autoencoder = Model(input_img, decoded) #784 -> 32 -> 784
    autoencoder.compile(loss="mse",optimizer=optimizer,metrics=["acc"])
    return autoencoder
# encoder = Model(input_img, encoded) # 754 -> 32



# encoded_input = Input(shape=(encoding_dim,))
# x = Dense(100, activation="relu")(encoded_input)
# x = Dense(50, activation="relu")(x)
# x = Dense(30, activation="relu")(x)
# decoder_layer = autoencoder.layers[-1]
# decoder = Model(encoded_input, decoder_layer(x)) #32->784

# autoencoder.summary()
# encoder.summary()
# decoder.summary()
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


model =KerasClassifier(build_fn = build_network_dnn, verbose=1)
def create_hyperparameters():
    batches=[5,7,9,10]
    optimizers = ["SGD", "adam"]
    dropout = np.linspace(0,0.1, 2)
    encoding_dim = [30,40,50,60]
    return{"batch_size":batches, "optimizer":optimizers,"keep_prob":dropout,"encoding_dim":encoding_dim}

# hyper = create_hyperparameters()
# search=RandomizedSearchCV(estimator=model, param_distributions=hyper, n_iter=10,n_jobs=-1, cv=4, verbose=1)


# autoencoder.compile(optimizer="adadelta",loss="categorical_crossentropy",metrics=["acc"])
model = build_network_dnn(optimizer = "adam",keep_prob=0,encoding_dim = 30)
history=model.fit(x_train, x_train,epochs=10, batch_size=500)

# print(search.best_params_)
# print("best score : ", search.best_score_)
# print("score:",search.score(x_test,x_test))
# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = autoencoder.predict_(x_test)

decoded_imgs = model.predict(x_test)
decoded_imgs[decoded_imgs<0]=0
# decoded_imgs[decoded_imgs>255]=255
print(decoded_imgs[0].reshape(32,32,3))
# print(encoded_imgs)
# print(decoded_imgs)
# print(encoded_imgs.shape)
# print(decoded_imgs.shape)


##########이미지출력#########
import matplotlib.pyplot as plt

n=10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(32,32,3))
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax=plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(32,32,3))
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


############그래프 출력#############
def plot_acc(history, title=None):
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history["acc"])
    plt.plot(history["val_acc"])
    if title is not None:
        plt.title(title)

    plt.ylabel("Aaccracy")
    plt.xlabel("epoch")
    plt.legend(["Training data", "validation data"], loc=0)

def plot_loss(history, title=None):
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    if title is not None:
        plt.title(title)

    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["Training data", "validation data"], loc=0)

plot_acc(history, "(a)학습 경과에 따른 정확도 변화 추이")
plt.show()
plot_loss(history, "(a)학습 경과에 따른 손실값 변화 추이")
plt.show()

loss, acc = autoencoder.evaluate(x_test, x_test)
print(loss, acc)

