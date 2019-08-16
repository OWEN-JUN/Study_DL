import tensorflow as tf
print(tf.__version__)

node1 = tf.constant([3.0,2.0],tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node1:",node1, "node2",node2)
print("node3:",node3)



### with
with tf.Session() as sess:
    print(sess.run(node1))
    print(sess.run(node2))
    print(sess.run(node3))