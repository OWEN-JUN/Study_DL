import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

mima = MinMaxScaler()
sta = StandardScaler()
tf.set_random_seed(777)
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973], 
                [823.02002, 828.070007, 1828100, 821.655029, 828.070007], 
                [819.929993, 824.400024, 1438100, 818.97998, 824.159973], 
                [816, 820.958984, 1008100, 815.48999, 819.23999], 
                [819.359985, 823, 1188100, 818.469971, 818.97998], 
                [819, 823, 1198100, 816, 820.450012], 
                [811.700012, 815.25, 1098100, 809.780029, 813.669983], 
                [809.51001, 816.659973, 1398100, 804.539978, 809.559998]]) 

x_train = xy[:,0:-1]
mima.fit(x_train[:,2:3])
sta.fit(x_train[:,:2])
x_train[:,2:3] = mima.transform(x_train[:,2:3])
x_train[:,:2] = sta.transform(x_train[:,:2])
sta.fit(x_train[:,3:])
x_train[:,3:] = sta.transform(x_train[:,3:])



y_train = xy[:,[-1]]
print("shape",x_train.shape, y_train.shape)
X = tf.placeholder(tf.float32,[None, 4])
Y = tf.placeholder(tf.float32,[None, 1])

w1 = tf.Variable(tf.random_normal([4,100]))
b1 = tf.Variable(tf.random_normal([100]))
l1 = tf.matmul(X, w1)+b1

w2 = tf.Variable(tf.random_normal([100,300]))
b2 = tf.Variable(tf.random_normal([300]))
l2 = tf.matmul(l1, w2)+b2

w3 = tf.Variable(tf.random_normal([300,50]))
b3 = tf.Variable(tf.random_normal([50]))
l3 = tf.matmul(l2, w3)+b3

w4 = tf.Variable(tf.random_normal([50,1]))
b4 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(l3, w4)+b4

# hypothesis = tf.matmul(X, w)+b
cost=tf.reduce_mean(tf.square(hypothesis - Y),axis=1)


#optimizer#################################################compile
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
train = tf.train.AdamOptimizer(learning_rate=0.2).minimize(cost)
# train = tf.train.AdadeltaOptimizer(learning_rate=0.5).minimize(cost)




with tf.Session() as sess:
    # with tf.device("/cpu:0"):
        # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(30001):
        _, cost_val, h_val = sess.run([train, cost, hypothesis], feed_dict={X: x_train, Y: y_train})
        
        if step % 500 == 0:
            print("Step: {:5}\nCost: {}\nh_val: {}".format(step, cost_val, h_val))
        if cost_val[0] < 0.004 and step > 20000 :
            break

    pred = sess.run([hypothesis], feed_dict={X: x_train})
    print("type : ",type(pred))
    pred = np.array(pred)
    y_train = y_train.reshape((-1,))
    pred = pred.reshape((-1,))
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.metrics import mean_absolute_error
    def RMSE(y_test, y_):
        return np.sqrt(mean_squared_error(y_test,y_))
    def RMAE(y_test, y_):
        return np.sqrt(mean_absolute_error(y_test,y_))
    print("RMSE:",RMSE(y_train,pred))
    print("RMAE:",RMAE(y_train,pred))
    

    print("r2:",r2_score(y_train,pred))

    
    
