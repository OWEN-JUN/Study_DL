import tensorflow as tf
import random
from keras.datasets import cifar10
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # one_hot 처리

# hyper parameters
learning_rate = 0.001
training_epochs = 30
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 28*28])
X_img = tf.reshape(X, [-1, 28, 28, 1])  # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 28, 28, 1)
# W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            # kernel_size: (3,3), channel: 1(흑백), output: 32
# print("W1: ", W1)   # shape=(3, 3, 1, 32)
#   Conv    ->   (?, 28, 28, 32)
#   Pool    ->   (?, 14, 14, 32)
# L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')  # stride: 몇 칸씩 움직일 것인가
# #                                       [    ] 가운데 값 두개만 주로 쓴다. 바깥 2개는 거의 고정
# # print("L1: ", L1)   # shape=(?, 28, 28, 32)
# L1 = tf.nn.relu(L1)
# L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], # (2, 2)로 자른 것을 2칸씩 이동 -> 반으로 줄어든다
#                       strides=[1, 2, 2, 1], padding='SAME')
# # print("L1: ", L1)   # shape=(?, 14, 14, 32)

# # L2 ImgIn shape=(?, 14, 14, 32)
# W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) # W1의 32
# # print("W2: ", W2)   # shape=(3, 3, 32, 64)
# #   Conv    ->   (?, 14, 14, 64)
# #   Pool    ->   (?, 7, 7, 64)
# L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
# # print("L2: ", L2)   # shape=(?, 14, 14, 64)
# L2 = tf.nn.relu(L2)
# L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
#                       strides=[1, 2, 2, 1], padding='SAME')
# # print("L2: ", L2)   # shape=(?, 7, 7, 64)   



L3 =tf.layers.conv2d(X_img,128,[3,3],activation=tf.nn.relu,padding="SAME")
L3 = tf.layers.max_pooling2d(L3,[2,2],[2,2])
L3 = tf.layers.dropout(L3,0.7)


# L4 =tf.layers.flatten(L3)
L4 = tf.layers.dense(L4,256,activation=tf.nn.relu)
L4 = tf.layers.dropout(L4,0.7)

L2_flat = tf.layers.dense(L4, 10, activation=tf.nn.relu)
print("L4:",L4)
# Final FC 7x7x64 inputs -> 10 outputs
W3 = tf.get_variable("W3", shape=[10, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat, W3) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model, Model Fit
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print('Label: ', sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))