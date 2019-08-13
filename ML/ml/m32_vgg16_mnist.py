from keras.applications import VGG16
from keras.applications import VGG19, Xception, InceptionV3,ResNet50, MobileNet
from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.datasets import mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
print(x_train.shape)
x_train = np.array(x_train).reshape((-1,np.prod(x_train.shape[1:])))
x_test = np.array(x_test).reshape((-1,np.prod(x_train.shape[1:])))
x_train=np.dstack([x_train] * 3)
x_test=np.dstack([x_test] * 3)
x_train = x_train.reshape(-1, 28,28,3)
x_test= x_test.reshape (-1,28,28,3)
print(x_train.shape)

from keras.preprocessing.image import img_to_array, array_to_img
x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_train])
x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_test])









y_train = np_utils.to_categorical(y_train)
y_test  = np_utils.to_categorical(y_test)
print(y_train)
model = Sequential()
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(48,48,3))
# conv_base = VGG16()
# conv_base2 = VGG19()

model.add(conv_base)

model.add(Flatten())
model.add(Dense(256))

model.add(Dense(10, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["acc"])
model.fit(x_train,y_train,epochs=100, batch_size=50)

print("acc: ",model.evaluate(x_test,y_test)[1])
