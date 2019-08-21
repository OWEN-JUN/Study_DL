import tensorflow as tf


class Model:
    # Model class => layer_Cat: 1 (회귀), 2(이진분류), 3(카테고리분류)
    # layer_num => 레이어 수
    

    class Dense:
        
        def __init__(self,output_dim,activate,input_dim=0,uplayer=None,keep=0,layer_cnt=0):
            self.layer_cnt = layer_cnt
            
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.uplayer = uplayer
            self.keep = keep
            self.activate = activate
            self.layer = None

        def make_layer(self):    
            print(self.layer_cnt)
            print(self.input_dim)
            print(self.output_dim)
            print("111111111111111111111111111")
            w = tf.get_variable("w%d"%(self.layer_cnt),shape=[self.input_dim, self.output_dim],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([self.output_dim]))
            if self.activate == "relu":
                layer = tf.nn.relu(tf.matmul(self.uplayer, w)+b)
            elif self.activate == "leaky_relu":
                layer = tf.nn.leaky_relu(tf.matmul(self.uplayer, w)+b)
        
            else: layer = tf.matmul(self.uplayer, w)+b

            if self.keep != 0:
                layer = tf.nn.dropout(layer, keep_prob=self.keep)
            
            self.layer = layer
            
            

        
    
    
        
    def __init__(self,X,layer_cat=0):
        self.layer_cat = layer_cat
        self.hidden_layer = []
        self.X = X

    #activate default : relu
    def add_dense(self,output_dim, input_dim = 0, activate="relu",keep=0):
        cnt = len(self.hidden_layer)
        uplayer = None
        if cnt == 0:
            uplayer = self.X
        else: 
            uplayer = self.hidden_layer[-1].layer
            input_dim = self.hidden_layer[-1].input_dim
            
            


     
        dense = Model.Dense(output_dim, input_dim=input_dim,uplayer=uplayer,activate=activate, keep=0,layer_cnt=cnt)    
        dense.make_layer()
        self.hidden_layer.append(dense)
        print(self.hidden_layer)

    def get_hypothesis(self):
        # print(self.hidden_layer)
        return self.hidden_layer[-1].layer
        
        





import tensorflow as tf
tf.set_random_seed(777) # 같은 랜덤 값

# X and Y data
x_train = [1., 2., 3.]
y_train = [1, 2, 3]


X = tf.placeholder(tf.float32)
m = Model(X)

m.add_dense(4,input_dim=1)
m.add_dense(1)

hypothesis = m.get_hypothesis()
#   constant와 placeholder 구분










# ■■■■■■■■■■ --- model.compile --- ■■■■■■■■■■
#   coss/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # (loss='mse', optimizer='adam')

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#   경사하강
# ■■■■■■■■■■ -------------------- ■■■■■■■■■■


# ■■■■■■■■■■ --- model.fit --- ■■■■■■■■■■
# Launch the graph in a session.
with tf.Session() as sess:  # with를 씀으로서 close 안해도 된다
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer()) # ★ 초기화(변수)

    # Fit the line
    for step in range(2001):    # epochs
        _, cost_val= sess.run([train, cost], feed_dict={X:x_train})   # sess.run은 fit이다
        #   _ 자리에 train 들어가고 순서대로

        if step % 20 == 0:
            print(step, cost_val)
# ■■■■■■■■■■ -------------------- ■■■■■■■■■■
            