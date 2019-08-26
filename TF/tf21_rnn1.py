import tensorflow as tf
import numpy as np

idx2char = ["e","h","i","l","o"]

# _data = np.array([["h","i","h","e","l","l","o"]]).reshape((-1.1))
_data = np.array([["h","i","h","e","l","l","o"]],dtype=np.str).reshape((-1,1))

print(_data.shape)
print(_data)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray().astype("float32") #알파벳 순서로 인코딩



print(_data)
print(_data.shape)

x_data = _data[:6,]
y_data = _data[1:,]
y_data = np.argmax(y_data, axis=1)
print("x_data",x_data)
print("y_data",y_data)
x_data = x_data.reshape(1,6,5)
y_data = y_data.reshape(1,6,)

print("x_data",x_data)
print("y_data",y_data)

num_classes = 5
batch_size = 1
sequence_length = 6

input_dim = 5
hidden_size = 5
learning_rate = 0.1

X = tf.compat.v1.placeholder(tf.float32, [None, sequence_length, input_dim])
Y = tf.compat.v1.placeholder(tf.int32, [None, sequence_length])
print(X)
print(Y)


#모델구성
cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

outputs,_states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

print(outputs)
print(outputs.shape)

X_for_fc = tf.reshape(outputs,[-1,hidden_size])
print(X_for_fc)
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])



weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
prediction = tf.argmax(outputs, axis=2)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        l, _ = sess.run([loss, train], feed_dict={X:x_data,Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})
        print(i, "loss:",l,"prediction: ",result,"true Y:",y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\nPrediction str : ",''.join(result_str))

