from keras.datasets import mnist
from keras.datasets import cifar10
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
# (x_train, _),(x_test, _) = mnist.load_data()
(x_train, _),(x_test, _) = cifar10.load_data()

# print(x_train[0])
minma = MinMaxScaler()
x_train = x_train.reshape((len(x_train),x_train.shape[1],x_train.shape[2],3))
x_train = x_train[:10]
x_train = np.array(x_train, dtype=np.float32)
# print(x_train.shape)
# print(x_train[0])



x_test = x_test.reshape((len(x_test),x_test.shape[1],x_test.shape[2],3))
print(x_train.shape)
print(x_test.shape)

x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

plt.figure(figsize=(1,1))
plt.imshow(x_train[0].reshape(32,32,3))
plt.show()
