import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]],dtype=np.float32)

# placeholdes for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_normal([2, 1]), name='weight')
b1 = tf.Variable(tf.random_normal([1]), name='bias')






# Hypothesis
hypothesis = tf.sigmoid(tf.matmul(X, W1) + b1)    # 0과 1 사이의 값

# cost/loss function 로지스틱 리그레션에서 cost에 -가 붙는다.
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

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