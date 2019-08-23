from keras.datasets import cifar10


import tensorflow as tf
import random
import numpy as np
def next_batch(num, data, labels):

#   `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.

    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# import matplotlib.pyplot as plt

tf.set_random_seed(777)

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
nb_classes = 10
X = tf.placeholder(tf.float32, [None, 32,32,3])

Y = tf.placeholder(tf.int32, [None, 1]) # 0 ~ 6

Y_one_hot = tf.one_hot(Y, nb_classes) # one-hot
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape one_hot:", Y_one_hot)
# y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
# y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10),axis=1)


learning_rate = 0.001

batch_size = 100

L1 =tf.layers.conv2d(X,64,[2,2],activation=tf.nn.relu,padding="SAME")
L1 = tf.layers.max_pooling2d(L1,[2,2],[2,2])
# L1 = tf.layers.dropout(L1,0.7)


L2 =tf.layers.conv2d(L1,128,[3,3],activation=tf.nn.relu,padding="SAME")
L2 = tf.layers.max_pooling2d(L2,[2,2],[2,2])
# L1 = tf.layers.dropout(L1,0.7)


L3 =tf.layers.conv2d(L2,256,[4,4],activation=tf.nn.relu)
L3 = tf.layers.max_pooling2d(L3,[2,2],[2,2])

W4 = tf.Variable(tf.random_normal([3, 3, 256, 500], stddev=0.01))
            # kernel_size: (3,3), channel: 1(흑백), output: 32
print("W4: ", W4)   # shape=(3, 3, 1, 32)
#   Conv    ->   (?, 28, 28, 32)
#   Pool    ->   (?, 14, 14, 32)
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')  # stride: 몇 칸씩 움직일 것인가
#                                       [    ] 가운데 값 두개만 주로 쓴다. 바깥 2개는 거의 고정
# print("L1: ", L1)   # shape=(?, 28, 28, 32)
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], # (2, 2)로 자른 것을 2칸씩 이동 -> 반으로 줄어든다
                      strides=[1, 2, 2, 1], padding='SAME')
# print("L1: ", L1)   # shape=(?, 14, 14, 32)


L5 =tf.layers.flatten(L4)
L5 = tf.layers.dense(L5,30,activation=tf.nn.relu)
# L3 = tf.layers.dropout(L3,0.7)

logits = tf.layers.dense(L5, 10, activation=tf.nn.relu)



# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=logits, labels=Y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=tf.stop_gradient([Y_one_hot])))

    #labels=tf.stop_gradient([Y_one_hot])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# hyper parameters
pre_ = tf.argmax(logits, 1)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    training_epochs = 5
    
    # train my model, Model Fit
    acc = 0
    while acc<0.7:
        print('Learning started. It takes sometime.')
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = x_train.shape[0] // batch_size
            for i in range(total_batch):
                batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)
                feed_dict = {X: batch_xs, Y: batch_ys}
                c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
                avg_cost += c / total_batch

            print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

        print('Learning Finished!')
        
        test_accuracy = 0.0  
        for i in range(10):
            batch_xs, batch_ys = next_batch(1000, x_test, y_test)
            test_accuracy = test_accuracy + accuracy.eval(feed_dict={X: batch_xs, Y: batch_ys})
        test_accuracy = test_accuracy / 10;
        print("테스트 데이터 정확도: %f" % test_accuracy)
        acc = test_accuracy
        
        # for i in range(10):
        #     print("pre_:",sess.run(pre_,feed_dict={X: x_test[i:i+1]}),"true:",y_test[i:i+1])




    




# Get one and predict
# import sys
# sys.exit()
# r = random.randint(0, len(x_test) - 1)
# print('Label: ', sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
# print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))