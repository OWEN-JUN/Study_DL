# Softmax Classifier
import tensorflow as tf
import pandas as pd
from keras.utils import np_utils
import numpy as np
tf.set_random_seed(777) # for reproducibility
# F:\Github\Study_DL\TF\data
zoo = pd.read_csv("./Study_DL/TF/data/data-04-zoo.csv",header = None)
zoo = np.array(zoo)
x_train = zoo[:,:-1]
y_train = zoo[:,-1]

y_train = np_utils.to_categorical(y_train,7)
print(y_train)
print(x_train.shape)
print(y_train.shape)
X = tf.placeholder("float", [None, 16])
Y = tf.placeholder("float", [None, 7])
nb_classes = 7

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')


hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1)) # categorical_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


predicted = tf.arg_max(hypothesis,1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.arg_max(Y,1)), dtype=tf.float32))
# Launch graph
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_train, Y: y_train})
        if step % 200 == 0:
            print(step, cost_val)

    print('■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
    # Testing & One-Hot encoding
    # a = sess.run(hypothesis, feed_dict={X: x_train})
    # print(a, sess.run(tf.argmax(a, 1)))
    h,p,a = sess.run([hypothesis,predicted, accuracy],
                        feed_dict={X: x_train, Y: y_train})
    for i in range(len(h)):
        print("\npredict:", int(p[i]),"true:",sess.run(tf.argmax(y_train,1)[i]) )
    print("\nAccuracy: ", a)
    
