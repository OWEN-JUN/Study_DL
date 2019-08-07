from keras.models import *
from keras.layers import Dense, LSTM, Dropout, Input, Conv2D, Flatten, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split, GridSearchCV
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],28,28,1).astype("float32")/255

x_test = x_test.reshape(x_test.shape[0],28,28,1).astype("float32")/255
print(y_train.shape)
y_train= np_utils.to_categorical(y_train)

y_test= np_utils.to_categorical(y_test)
print(y_train.shape)

def build_network_cnn(keep_prob=0.5, optimizer="adam"):
    inputs = Input(shape=(28,28,1), name="input")
    x = Conv2D(32,kernel_size=(3,3),activation="relu")(inputs)
    x = Dropout(keep_prob)(x)
    x = Conv2D(64,(3,3),activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(keep_prob)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu", name="hidden2")(x)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation="relu", name="hidden3")(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(10, activation="softmax", name="output")(x)
    model = Model(input = inputs, outputs=prediction)
    model.compile(optimizer = optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
    

def create_hyperparameters():
    batches=[10,20,30,40,50]
    optimizers = ["rmsprop", "adam", "adadelta"]
    dropout = np.linspace(0.1,0.5, 5)
   
    return[{"batch_size":batches, "optimizer":optimizers,"keep_prob":dropout}]

from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn = build_network_cnn, verbose=1)

hyperparameters=create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import  GridSearchCV
# search=RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=10, n_jobs=1, cv=3, verbose=1)
search=GridSearchCV(estimator=model, param_grid=hyperparameters,cv=3)

search.fit(x_train, y_train)
print(search.best_params_)