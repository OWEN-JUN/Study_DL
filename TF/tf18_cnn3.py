import tensorflow as tf
import random
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # one_hot 처리

# hyper parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 64

# input place holders
X = tf.placeholder(tf.float32, [None, 28*28])
X_img = tf.reshape(X, [-1, 28, 28, 1])  # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            # kernel_size: (3,3), channel: 1(흑백), output: 32
# print("W1: ", W1)   # shape=(3, 3, 1, 32)
#   Conv    ->   (?, 28, 28, 32)
#   Pool    ->   (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')  # stride: 몇 칸씩 움직일 것인가
#                                       [    ] 가운데 값 두개만 주로 쓴다. 바깥 2개는 거의 고정
# print("L1: ", L1)   # shape=(?, 28, 28, 32)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], # (2, 2)로 자른 것을 2칸씩 이동 -> 반으로 줄어든다
                      strides=[1, 2, 2, 1], padding='SAME')
# print("L1: ", L1)   # shape=(?, 14, 14, 32)

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 80], stddev=0.01)) # W1의 32
# print("W2: ", W2)   # shape=(3, 3, 32, 64)
#   Conv    ->   (?, 14, 14, 64)
#   Pool    ->   (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
# print("L2: ", L2)   # shape=(?, 14, 14, 64)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                      strides=[1, 1, 1, 1], padding='SAME')
# print("L2: ", L2)   # shape=(?, 7, 7, 64)

# L3 ImgIn shape=(?, 7, 7, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 80, 200], stddev=0.01))
# print("W3: ", W3)   # shape=(3, 3, 64, 32)
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
print("L3: ", L3)   # shape=(?, 7, 7, 32)

# L4 ImgIn shape=(?, 7, 7, 32)
W4 = tf.Variable(tf.random_normal([3, 3, 200, 300], stddev=0.01))
# print("W4: ",  W4)  # shape=(3, 3, 32, 16)
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1],
                      strides=[1, 1, 1, 1], padding='SAME')
print("L4: ", L4)   # shape=(?, 4, 4, 32)


L4_flat = tf.reshape(L4, [-1, 14 * 14 * 300])

#
W5 = tf.get_variable("W5", shape=[14 * 14 * 300, 20],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([20]))
logits = tf.matmul(L4_flat, W5) + b5

# Final FC 7x7x64 inputs -> 10 outputs
W6 = tf.get_variable("W6", shape=[20 , 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(logits, W6) + b6

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# hyper parameters
learning_rate = 0.001
training_epochs = 3
batch_size = 300

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = 0
while acc < 0.994:
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
    acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    print('Accuracy:', acc)


# Test model and check accuracy

print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print('Label: ', sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))