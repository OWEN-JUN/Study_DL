import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

x_data=[[1,2,1],[1,3,2],[1,3,4],[1,5,5],[1,7,5],[1,2,5],[1,6,6],[1,7,7]]

y_data=[[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]


x_test = [[2,1,1],[3,1,2],[3,3,4]]
y_test = [[0,0,1],[0,0,1],[0,0,1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

w = tf.Variable(tf.random_normal([3,3]))
b = tf.Variable(tf.random_normal([3]))


hypothesis = tf.nn.softmax(tf.matmul(X, w)+b)

#cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))

# try to change learnin_rate to small numbers
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

#correct prediction test model
prediction = tf.argmax(hypothesis,1)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(1001):
        _, cost_val, w_val = sess.run([optimizer, cost, w], feed_dict={X: x_data, Y: y_data})
        
        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\nw_val: {}".format(step, cost_val, w_val))

    # Let's see if we can predict
    a, pred = sess.run([accuracy, prediction], feed_dict={X: x_test,Y: y_test})
    # y_data: (N, 1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, np.array(y_data)):
        print("Prediction: {} True Y: {}".format(p, np.argmax(y)))
    print("acc :",a)


#### lr =0.01
# Prediction: 2 True Y: 2
# Prediction: 2 True Y: 2
# Prediction: 2 True Y: 2
# acc : 1.0


#### lr = 0.01
# Prediction: 0 True Y: 2
# Prediction: 0 True Y: 2
# Prediction: 1 True Y: 2
# acc : 0.0


#### lr = 0.006
# Prediction: 2 True Y: 2
# Prediction: 1 True Y: 2
# Prediction: 1 True Y: 2
# acc : 0.33333334


#### lr = 1
# Prediction: 2 True Y: 2
# Prediction: 2 True Y: 2
# Prediction: 2 True Y: 2
# acc : 1.0


#### lr = 10
# Prediction: 0 True Y: 2
# Prediction: 0 True Y: 2
# Prediction: 0 True Y: 2
# acc : 0.0