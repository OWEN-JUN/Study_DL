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


def layer(input, output,uplayer,dropout=0,end=False):
    w = tf.Variable(tf.random_normal([input,output]))
    b = tf.Variable(tf.random_normal([output]))
    if ~end:
        layer = tf.sigmoid(tf.matmul(uplayer, w)+b)
    else: layer = tf.matmul(uplayer, w)+b

    if dropout != 0:
        layer = tf.nn.dropout(layer, dropout)

    return layer

X = tf.placeholder(tf.float32,[None, 784])
Y = tf.placeholder(tf.float32,[None, 10])




l1 = layer(784,256,X)
l2 = layer(256,256,l1)
# l3 = layer(500,100,l2,dropout=0.2)
# l4 = layer(100,50,l3,dropout=0.2)
# l5 = layer(50,30,l4,dropout=0.2)
# l6 = layer(30,20,l5,dropout=0.2)
logits = layer(256,10,l2,end=True)


hypothesis = tf.nn.softmax(logits)
# cross entropy cost/loss
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(logits), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = hypothesis, labels=Y))

# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
# train = tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(logits, 1)


is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y,1))

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


num_epochs = 50
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
        print("accuracy:", accuracy.eval(session=sess, feed_dict={X:batch_xs, Y:batch_ys}))
        
    writer = tf.summary.FileWriter('D:/board', sess.graph)
    print("learning finished")

    print("accuracy:", accuracy.eval(session=sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

    r = random.randint(0,mnist.test.num_examples - 1)
    print("label:", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))

    print("prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X:mnist.test.images[r:r+1]}))


    plt.imshow(
        mnist.test.images[r:r+1].reshape(28,28),
        cmap="Greys",
        interpolation="nearest"
    )

    plt.show()

