import tensorflow as tf
tf.set_random_seed(777)
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Model:
    # Model class => layer_Cat: 1 (회귀), 2(이진분류), 3(카테고리분류)
    # layer_num => 레이어 수
    def __init__(self, layer_cat, X):
        self.layer_cat = layer_cat
        self.hidden_layer = []
        self.X = X

    class Dense:
        
        def __init__(self,output_dim,activate,input_dim=0,uplayer=None,keep=0,layer_cnt=None):
            self.layer_cnt = layer_cnt
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.uplayer = uplayer
            self.keep = keep
            self.activate = activate

        def make_layer(self):    
           
            w = tf.get_variable("w%d"%(self.layer_cnt),shape=[self.input_dim, self.output_dim],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([self.output_dim]))
            if self.activate == "relu":
                layer = tf.nn.relu(tf.matmul(self.uplayer, w)+b)
            elif self.activate == "leaky_relu":
                layer = tf.nn.leaky_relu(tf.matmul(self.uplayer, w)+b)
        
            else: layer = tf.matmul(self.uplayer, w)+b

            if self.keep != 0:
                layer = tf.nn.dropout(layer, keep_prob=self.keep)
            Model.hidden_layer.append(layer)
            return layer

        def connect_layer(self):
            cnt = len(Model.hidden_layer)
            uplayer = None
            if cnt == 0:
                self.uplayer = self.X
            else: 
                self.uplayer = Model.hidden_layer[-1]
                self.input_dim = Model.hidden_layer[-1].shape[1]
            make_layer()

    
    
        


    # #activate default : relu
    # def add_(Dense(self, output_dim, input_dim, keep=0)):
    #     connect_layer() 
        
            
        


    














boston_housing_x = np.load("./Study_DL/TF/data/boston_housing_x.npy")
boston_housing_y = np.load("./Study_DL/TF/data/boston_housing_y.npy")
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


cnt = 1
def layer(input, output,uplayer,dropout=0,end=False):
    global cnt
    w = tf.get_variable("w%d"%(cnt),shape=[input, output],initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([output]))
    if ~end:
        layer = tf.nn.leaky_relu(tf.matmul(uplayer, w)+b)
    else: layer = tf.matmul(uplayer, w)+b

    if dropout != 0:
        layer = tf.nn.dropout(layer, keep_prob=dropout)
    cnt += 1
    
    return layer


keep_prob = 0



l1 = layer(13,50,X,dropout=0.8)
l2 = layer(50,100,l1,dropout=0.8)

print("l1l1l1l1l1l1l1",l1.shape[1])
hypothesis = layer(100,1,l2,end=True)

import sys
sys.exit()



#cost/loss function#################################################
cost=tf.reduce_mean(tf.square(hypothesis - Y))


#optimizer#################################################compile
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
train = tf.train.AdamOptimizer(learning_rate=0.006).minimize(cost)
# train = tf.train.AdadeltaOptimizer(learning_rate=0.05,).minimize(cost)



#launch the graph in a session#################################################  model fit
feed_dict={X:x_train,Y:y_train}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #### model learning#################################################
    for step in range(10501):
        # _, cost_val, w_val, b_val = sess.run([train, cost, w, b], feed_dict=feed_dict)
        _, cost_val = sess.run([train, cost],feed_dict = feed_dict)

        if step%500 == 0:
            print(step, cost_val)
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
