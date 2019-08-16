import tensorflow as tf
print(tf.__version__)

##placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b
add_and_triple = adder_node *3


print(a)
print(b)
print(adder_node)
print(add_and_triple)
### with
with tf.Session() as sess:
    print(sess.run(node1))
    print(sess.run(node2))
    print(sess.run(node3))
    print(sess.run(adder_node, feed_dict={a:3,b:4.5}))
    print(sess.run(adder_node, feed_dict={a:[1,3],b:[2,4]}))
    print(sess.run(add_and_triple, feed_dict={a:3,b:4.5}))
 




