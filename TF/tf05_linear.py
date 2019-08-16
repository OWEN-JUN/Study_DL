import tensorflow as tf
tf.set_random_seed(777)
import numpy as np
x = np.array([1,2,3]) ##train_set
y = np.array([1,2,3]) ##train_y_set



x_train = tf.placeholder(tf.float32)
w = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# hypothesis = x_train * w + b
hypothesis = x * w + b



#cost/loss function#################################################
cost=tf.reduce_mean(tf.square(hypothesis - y))


#optimizer#################################################
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
train = tf.train.AdamOptimizer(learning_rate=0.3).minimize(cost)



#launch the graph in a session#################################################
feed_dict={x_train:x}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #### model learning#################################################
    for step in range(1001):
        # _, cost_val, w_val, b_val = sess.run([train, cost, w, b], feed_dict=feed_dict)
        _, cost_val, w_val, b_val = sess.run([train, cost, w, b])

        if step%20 == 0:
            print(step, cost_val, w_val, b_val)
    #### model predict#################################################
    # print(sess.run(hypothesis, feed_dict=feed_dict))
    print(sess.run(hypothesis))

