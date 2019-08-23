

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


tf.set_random_seed(777)


boston_housing_x = np.load("./data/boston_housing_x.npy")
boston_housing_y = np.load("./data/boston_housing_y.npy")
# iris_label = np.load("./Study_DL/TF/data/iris_label.npy")


sta = StandardScaler()
sta.fit(boston_housing_x)
boston_housing_x = sta.transform(boston_housing_x)
x_train = boston_housing_x

y_train = boston_housing_y.reshape((-1,1))
# y_train = np.array(y_train,dtype=np.int32)




print(x_train.shape, y_train.shape)
x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.2)
print(x_train.shape, y_train.shape)




X = tf.placeholder(tf.float32,[None, 13])
Y = tf.placeholder(tf.float32,[None, 1])


l1 = tf.layers.dense(X, 100, activation=tf.nn.relu)
l2 = tf.layers.dense(l1, 200, activation=tf.nn.relu)
l3 = tf.layers.dense(l2, 100, activation=tf.nn.relu)
hypothesis = tf.layers.dense(l3, 1, activation=tf.nn.relu)




#cost/loss function#################################################
cost=tf.reduce_mean(tf.square(hypothesis - Y))


#optimizer#################################################compile
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
# train = tf.train.AdadeltaOptimizer(learning_rate=0.05,).minimize(cost)



#launch the graph in a session#################################################  model fit
feed_dict={X:x_train,Y:y_train}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #### model learning#################################################
    for step in range(8501):
        # _, cost_val, w_val, b_val = sess.run([train, cost, w, b], feed_dict=feed_dict)
        _, cost_val = sess.run([train, cost],feed_dict = feed_dict)

        if step%500 == 0:
            print("Step: {:5}\tCost: {:f}".format(step, cost_val))
            
    #### model predict#################################################
    # print(sess.run(hypothesis, feed_dict=feed_dict))
    pred = sess.run(hypothesis,feed_dict = {X:x_test})

    pred = np.array(pred)
    y_test = y_test.reshape((-1,))
    pred = pred.reshape((-1,))
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.metrics import mean_absolute_error
    def RMSE(y_test, y_):
        return np.sqrt(mean_squared_error(y_test,y_))
    def RMAE(y_test, y_):
        return np.sqrt(mean_absolute_error(y_test,y_))
    print("RMSE:",RMSE(y_test,pred))
    print("RMAE:",RMAE(y_test,pred))
    

    print("r2:",r2_score(y_test,pred))
