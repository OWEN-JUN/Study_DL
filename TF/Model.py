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
        self.optimizer=None
        self.loss = None
        self.metrics = None
        self.lr = 0.001
        self.hypothesis = None
        self.sess = None

    #activate default : relu
    def add_dense(self,output_dim, input_dim = 0, activation="relu",keep=0):
        cnt = len(self.hidden_layer)
        uplayer = None
        if cnt == 0:
            uplayer = self.X
        else: 
            uplayer = self.hidden_layer[-1].layer
            input_dim = self.hidden_layer[-1].output_dim            
            


     
        dense = Model.Dense(output_dim, input_dim=input_dim,uplayer=uplayer,activate=activation, keep=0,layer_cnt=(cnt+1))    
        dense.make_layer()
        self.hidden_layer.append(dense)
        print(self.hidden_layer)

    def get_hypothesis(self):
        # print(self.hidden_layer)
        
        return self.hidden_layer[-1].layer




    def compile(self, loss, optimizer,lr=0.001):
        loss_list = ["mse", "binary_crossentropy","categorical_crossentropy"]
        opti_list = ["adam","adadelta","grd"]
        metrics_list = ["mse","rmse","r2","rmae"]

        self.loss = loss_list.index(loss.lower())
        self.optimizer= opti_list.index(optimizer.lower())
        # self.metrics = metrics_list.index(metrics.lower())
        self.lr = lr

        
    def fit(self, x, y, epochs, batch_size=30):
        hypothesis = self.get_hypothesis()
        feed_dict = {X:x, Y:y}
        if self.loss == 0:
            cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # (loss='mse')
        elif self.loss == 1:
            cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis)) #binary_crossentropy
        elif self.loss == 2:
            cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1)) # categorical_crossentropy



        if self.optimizer ==0:
            train = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)
        elif self.optimizer ==1:
            train = tf.train.AdadeltaOptimizer(learning_rate=self.lr).minimize(cost)
        elif self.optimizer ==2:
            train = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(cost)
        self.sess = tf.Session()
        # with tf.Session() as sess:  # with를 씀으로서 close 안해도 된다
            # Initializes global variables in the graph.
        self.sess.run(tf.global_variables_initializer()) # ★ 초기화(변수)

            # Fit the line
        for step in range(epochs):    # epochs
            _, cost_val= self.sess.run([train, cost], feed_dict=feed_dict)   # sess.run은 fit이다
            #   _ 자리에 train 들어가고 순서대로

            if step % 20 == 0:
                print("epoch : ",step,"cost : ",cost_val)
        self.hypothesis = hypothesis


    def predict(self, x):
        hypothesis = self.hypothesis
        feed_dict={X:x}
        if self.loss == 0:
            hypothesis = hypothesis


        
        print(self.sess.run([hypothesis],feed_dict=feed_dict ))
        self.sess.close()

        

        
        





import tensorflow as tf
tf.set_random_seed(777) # 같은 랜덤 값

# X and Y data
x_train = [[1.], [2.], [3.]]
y_train = [[1], [2], [3]]


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


m = Model(X)

m.add_dense(1,input_dim=1)
# m.add_dense(1)


m.compile(loss="mse",optimizer="adam")
m.fit(x_train, y_train, epochs=1000)
m.predict(x_train)