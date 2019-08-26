from keras.applications import VGG16
from keras.applications import VGG19, Xception, InceptionV3,ResNet50, MobileNet
from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.datasets import mnist
from keras.datasets import cifar10

# (x_train, y_train),(x_test, y_test) = mnist.load_data()
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

# x_train = x_train[:35000]
# y_train = y_train[:35000]
print(x_train.shape)

# x_train = x_train.astype("float32")/255
# x_test = x_test.astype("float32")/255
# x_train = np.array(x_train).reshape((-1,np.prod(x_train.shape[1:])))
# x_test = np.array(x_test).reshape((-1,np.prod(x_test.shape[1:])))

# x_train = x_train.reshape(-1, 56,56,3)
# x_test= x_test.reshape (-1,56,56,3)
print(x_train.shape)


# x_train = x_train.astype("float32")/255
# x_test = x_test.astype("float32")/255
from keras.preprocessing.image import img_to_array, array_to_img
# x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_train])
# x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_test])


print(x_train.shape)


y_train = np_utils.to_categorical(y_train)
y_test  = np_utils.to_categorical(y_test)
print(y_train)
model = Sequential()
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(32,32,3))
# conv_base = VGG16()
# conv_base2 = VGG19()

model.add(conv_base)

model.add(Flatten())
model.add(Dense(50))

model.add(Dense(10, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adadelta",metrics=["acc"])
model.fit(x_train,y_train,epochs=300, batch_size=1500)

print("acc: ",model.evaluate(x_test,y_test)[1])
