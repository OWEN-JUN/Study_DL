import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]],dtype=np.float32)

# placeholdes for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_normal([2, 100]), name='weight')
b1 = tf.Variable(tf.random_normal([100]), name='bias')
l1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([100, 80]), name='weight')
b2 = tf.Variable(tf.random_normal([80]), name='bias')
l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)

W3 = tf.Variable(tf.random_normal([80, 70]), name='weight')
b3 = tf.Variable(tf.random_normal([70]), name='bias')
l3 = tf.nn.relu(tf.matmul(l2, W3) + b3)

W4 = tf.Variable(tf.random_normal([70, 60]), name='weight')
b4 = tf.Variable(tf.random_normal([60]), name='bias')
l4 = tf.nn.relu(tf.matmul(l3, W4) + b4)

W5 = tf.Variable(tf.random_normal([60, 50]), name='weight')
b5 = tf.Variable(tf.random_normal([50]), name='bias')
l5 = tf.nn.relu(tf.matmul(l4, W5) + b5)

W6 = tf.Variable(tf.random_normal([50, 40]), name='weight')
b6 = tf.Variable(tf.random_normal([40]), name='bias')
l6 = tf.nn.relu(tf.matmul(l5, W6) + b6)

W7 = tf.Variable(tf.random_normal([40, 30]), name='weight')
b7 = tf.Variable(tf.random_normal([30]), name='bias')
l7 = tf.nn.relu(tf.matmul(l6, W7) + b7)

W8 = tf.Variable(tf.random_normal([30, 20]), name='weight')
b8 = tf.Variable(tf.random_normal([20]), name='bias')
l8 = tf.sigmoid(tf.matmul(l7, W8) + b8)

W9 = tf.Variable(tf.random_normal([20, 10]), name='weight')
b9 = tf.Variable(tf.random_normal([10]), name='bias')
l9 = tf.sigmoid(tf.matmul(l8, W9) + b9)

W0 = tf.Variable(tf.random_normal([10, 5]), name='weight')
b0 = tf.Variable(tf.random_normal([5]), name='bias')
le = tf.sigmoid(tf.matmul(l9, W0) + b0)




We = tf.Variable(tf.random_normal([5, 1]), name='weight')
be = tf.Variable(tf.random_normal([1]), name='bias')




# Hypothesis
hypothesis = tf.sigmoid(tf.matmul(le, We) + be)    # 0과 1 사이의 값

# cost/loss function 로지스틱 리그레션에서 cost에 -가 붙는다.
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

# Accuracy computatiom
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))    # 일반적인 선형 회귀에선 안된다

# Launch graph
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis:\n", c, "\nCorrect (Y):\n", y_data, "\nAccuracy: ", a)