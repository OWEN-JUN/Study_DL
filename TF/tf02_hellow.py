import tensorflow as tf
print(tf.__version__)

hello = tf.constant("Hello world")

###
sess = tf.Session()
print(sess.run(hello))

### with
with tf.Session() as sess:
    print(sess.run(hello))