print("aaaa")

a = 5
b = 10
print(a+b)

import tensorflow as tf
import keras
ten = tf.constant(5)
inp = tf.placeholder(tf.float32)
feed = {inp : 3}
print(ten)

with tf.Session() as sess:
    print(sess.run(ten))
    print(sess.run(inp, feed_dict = feed))