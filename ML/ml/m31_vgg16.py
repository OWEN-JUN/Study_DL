from keras.applications import VGG16
from keras.applications import VGG19, Xception, InceptionV3,ResNet50, MobileNet
from keras.models import *
from keras.layers import *
model = Sequential()
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))
# conv_base = VGG16()
# conv_base2 = VGG19()

model.add(conv_base)

# model.add(Flatten())
model.add(Dense(256))

model.add(Dense(1))
model.summary()
