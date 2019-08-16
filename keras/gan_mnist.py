from keras.models import *
from keras.layers import *
from keras.datasets import mnist
from keras.optimizers import RMSprop
import numpy as np


import matplotlib.pyplot as plt
import keras
from tqdm import tqdm


(x_train, y_train),(x_test, y_test) = mnist.load_data()
img_row=28
img_col=28
channel=1
batch_size=10
x_train = x_train.reshape((-1,28,28,1))
# 식별자가 실제 이미지를 인식하는 방법에는 Figure 1에서 보이는 바와 같이 심화 컨볼루션 신경망(deep Convolutional Neural Network, 이하 DCNN)이 기본적으로 사용됩니다.
# MNIST 데이터셋으로 28 픽셀x 28 픽셀x 1 채널의 이미지를 입력 데이터로 합니다.
# 시그모이드(sigmoid) 결과는 실제 이미지 정도의 확률을 (0.0이 가짝 1.0이 진짜, 회색 영역은 임의의 값)과 같이 단일 값으로 보여줍니다.
# 전형적인 CNN과 다른 점은 레이어 사이에서 최대 풀링(max-pooling)이 없다는 것입니다. 대신에 다운 샘플링(downsampling, 임의로 이미지의 일부를 샘플링 하는 것) 방식을 취합니다.
# 각 CNN 레이어의 활성함수(activation function)는 ReLU를 사용합니다.
# 레이어 사이에 0.4~0.7의 값을 가지도록 드랍아웃(dropout) 값을 설정하면 과적합(over fitting)과 암기(특정 값을 외우는 것, memorization)를 방지할 수 있습니다.

modeld = Sequential()
depth = 64
dropout = 0.4

input_shape = (img_row, img_col, channel)
modeld.add(Conv2D(depth*1,5, strides = 2, input_shape = input_shape, padding="same", activation="relu"))
modeld.add(Dropout(dropout))
modeld.add(Conv2D(depth*2, 5,strides = 2, padding="same", activation="relu"))
modeld.add(Dropout(dropout))
modeld.add(Conv2D(depth*4, 5,strides = 2, padding="same", activation="relu"))
modeld.add(Dropout(dropout))
modeld.add(Conv2D(depth*8, 5,strides = 2, padding="same", activation="relu"))
modeld.add(Dropout(dropout))

modeld.add(Flatten())
modeld.add(Dense(1, activation="sigmoid"))
modeld.compile(loss='binary_crossentropy', optimizer="adam")


######generator###########
modelg = Sequential()
dropout = 0.4
depth = 64+64+64+64
dim = 7

modelg.add(Dense(dim*dim*depth, input_dim=100, activation="relu"))
modelg.add(BatchNormalization(momentum=0.9))
modelg.add(Reshape((dim,dim,depth)))
modelg.add(Dropout(dropout))


modelg.add(UpSampling2D())
modelg.add(Conv2DTranspose(int(depth/2),5,padding="same", activation="relu"))
modelg.add(BatchNormalization(momentum=0.9))
modelg.add(UpSampling2D())
modelg.add(Conv2DTranspose(int(depth/4),5,padding="same", activation="relu"))
modelg.add(BatchNormalization(momentum=0.9))

modelg.add(Conv2DTranspose(int(depth/8),5,padding="same", activation="relu"))
modelg.add(BatchNormalization(momentum=0.9))
modelg.add(Conv2DTranspose(1,5,padding="same", activation="sigmoid"))

modelg.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

##생성+식별자 모델

optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
gd = Sequential()
gd.add(modelg)
gd.add(modeld)
gd.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["acc"])


img_train = x_train[np.random.randint(0,x_train.shape[0], size=batch_size),:,:,:]
noise = np.random.uniform(-1.0,1.0,size=[batch_size, 100])
img_fake = modelg.predict(noise)
print(img_train.shape)
print(img_fake.shape)

x = np.concatenate((img_train, img_fake))
y = np.ones(([2*batch_size, 1]))
y[batch_size:,:] = 0
d_loss= modeld.train_on_batch(x,y)
y = np.ones([batch_size,1])
noise = np.random.uniform(-1.0,1.0,size=[batch_size, 100])
a_loss = gd.train_on_batch(noise, y)



def training(epochs=1, batch_size=128):
    
    #Loading the data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    batch_count = X_train.shape[0] / batch_size
    X_train = X_train.reshape((-1,28,28,1))
   
    
    ## structure of model
    # from keras.utils import plot_model
    # plot_model(modelg, show_shapes=True, to_file='generator.png')
    
    # plot_model(modeld, show_shapes=True, to_file='discriminator.png')
    
    # plot_model(gd, show_shapes=True, to_file='gan.png')
    
    for e in range(1,epochs+1):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
        # Generate random noise as an input to initialize the generator
            noise= np.random.normal(0,1, [batch_size, 100])
            
            # Generate fake MNIST images from noised input
            generated_images = modelg.predict(noise)
            
            # Get a random set of  real images
            image_batch =X_train[np.random.randint(0,X_train.shape[0], size=batch_size)]
            # print(generated_images.shape)
            # print(image_batch.shape)
            # [np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
            # Construct different batches of real and fake data 
            X= np.concatenate([image_batch, generated_images])
            
            # Labels for generated and real data
            y_dis=np.zeros(2*batch_size)
            y_dis[:batch_size]=0.9
            
            # Pretrain discriminator on  fake and real data before starting the gan. 
            modeld.trainable=True
            modeld.train_on_batch(X, y_dis)
            
            # Tricking the noised input of the Generator as real data
            y_gen = np.ones(batch_size)
            
            # During the training of gan, the weights of discriminator should be fixed. 
            # We can enforce that by setting the trainable flag
            modeld.trainable=False
            
            # Training  the GAN by alternating the training of the Discriminator and training the chained GAN model with Discriminator's weights freezed.
            gd.train_on_batch(noise, y_gen)
            
        if e % 100 == 0:
            
            n=10
            plt.figure(figsize=(20,4))
            for i in range(n):
                ax = plt.subplot(2,n,i+1)
                plt.imshow(image_batch[i].reshape(28,28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                ax=plt.subplot(2,n,i+1+n)
                plt.imshow(generated_images[i].reshape(28,28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            plt.show()
            # plot_generated_images(e, modelg)

training(10000,50)