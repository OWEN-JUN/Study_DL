import tensorflow as tf
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

lr = 0.001
epoch = 15
batch_size = 100


X = tf.placeholder(tf.float32,[None, 784])
X_IMG = tf.reshape(X, [-1,28,28,1])
Y = tf.placeholder(tf.float32, [None, 10])

w1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
print("w1",w1)
l1 = tf.nn.conv2d(X_IMG, w1, strides=[1,1,1,1], padding="SAME")
print("l1",l1)
l1 = tf.nn.relu(l1)
l1 = tf.nn.max_pool(l1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
print("l1",l1)



w2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))

l2 = tf.nn.conv2d(l1, w2, strides=[1,1,1,1], padding="SAME")
print("l2",l2)
l2 = tf.nn.relu(l2)
l2 = tf.nn.max_pool(l2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
print("l2",l2)
l2_flat = tf.reshape(l2, [-1, 7*7*64])
