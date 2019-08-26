
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
def make_random_data():
    x = np.random.uniform(low=-2, high = 2, size=200)
    y=[]
    for t in x:
        r = np.random.normal(loc=0.0, scale=(0.5+t*t/3),size=None)
        y.append(r)

    return x, 1.726*x - 0.84 +np.array(y)

x, y = make_random_data()
# plt.plot(x,y,"o")
# plt.show()

import tensorflow.compat.v1 as tf

x_train, y_train = x[:150],y[:150]
x_test, y_test = x[150:],y[150:]
tf.keras.optimizers.SGD(lr=0.0000001)

dkdk


input = tf.keras.Input(shape=(1,))
output = tf.keras.layers.Dense(1)(input)

model = tf.keras.Model(input, output)



model.summary()

model.compile(optimizer="sgd",loss="mse")
history = model.fit(x_train,y_train, epochs=300, validation_split=0.3)
epochs = np.arange(1,300+1)
plt.plot(epochs, history.history["loss"],label="training loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()





