from keras.applications import VGG16
from keras.applications import VGG19, Xception, InceptionV3,ResNet50, MobileNet
from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.datasets import mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train[:50]
y_train = y_train[:50]
print(x_train.shape)
# x_train = x_train.astype("float32")/255
# x_test = x_test.astype("float32")/255
x_train = np.array(x_train).reshape((-1,np.prod(x_train.shape[1:])))
x_test = np.array(x_test).reshape((-1,np.prod(x_train.shape[1:])))
x_train=np.dstack([x_train] * 3)
x_test=np.dstack([x_test] * 3)
x_train = x_train.reshape(-1, 28,28,3)
x_test= x_test.reshape (-1,28,28,3)
print(x_train.shape)
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
from keras.preprocessing.image import img_to_array, array_to_img
x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((75,75))) for im in x_train])
x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((75,75))) for im in x_test])


print(x_train.shape)


y_train = np_utils.to_categorical(y_train)
y_test  = np_utils.to_categorical(y_test)
print(y_train)
model = Sequential()
# conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(48,48,3))
# conv_base = VGG19(weights="imagenet", include_top=False, input_shape=(48,48,3))
conv_base = ResNet50(weights="imagenet", include_top=False, input_shape=(75,75,3))


# conv_base = VGG16()
# conv_base2 = VGG19()

model.add(conv_base)

model.add(Flatten())
# model.add(Dense(50))

model.add(Dense(10, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adadelta",metrics=["acc"])
model.fit(x_train,y_train,epochs=10, batch_size=600)

print("acc: ",model.evaluate(x_test,y_test)[1])
