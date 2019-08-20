import tensorflow as tf
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

print(mnist.train.images)
print(mnist.test.images)
print(mnist.train.images.shape)
print(mnist.test.images.shape)
print(mnist.train.labels)


###
#코딩하시오 X,Y,w,b, hypothesis, cost, train
###



X = tf.placeholder(tf.float32,[None, 784])
Y = tf.placeholder(tf.float32,[None, 10])

# w1 = tf.Variable(tf.random_normal([784,100]))
# b1 = tf.Variable(tf.random_normal([100]))
# l1 = tf.matmul(X, w1)+b1

# w2 = tf.Variable(tf.random_normal([1000,5000]))
# b2 = tf.Variable(tf.random_normal([5000]))
# l2 = tf.matmul(l1, w2)+b2

# w3 = tf.Variable(tf.random_normal([5000,500]))
# b3 = tf.Variable(tf.random_normal([500]))
# l3 = tf.matmul(l2, w3)+b3

w4 = tf.Variable(tf.random_normal([784,10]))
b4 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(X, w4)+b4



hypothesis = tf.nn.softmax(logits)
# cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
# train = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(cost)
# train = tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)


is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


num_epochs = 300
batch_size = 100
num_iterations = int(mnist.train.num_examples/batch_size)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([train, cost], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += (cost_val / num_iterations)
        print(avg_cost)
        print("epoch: {:04d}, cost: {:.9f}".format(epoch + 1, avg_cost))
        if avg_cost < 0.01:
            break

    print("learning finished")

    print("accuracy:", accuracy.eval(session=sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

    r = random.randint(0,mnist.test.num_examples - 1)
    print("label:", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))

    print("prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1]}))


    plt.imshow(
        mnist.test.images[r:r+1].reshape(28,28),
        cmap="Greys",
        interpolation="nearest"
    )

    plt.show()

