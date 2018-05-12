from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
from keras.callbacks import TensorBoard

#prep the data
(x_train,_) , (x_test,_) = mnist.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = x_train.reshape((len(x_train),28,28,1))
x_test = x_test.reshape((len(x_test),28,28,1))
print (x_train.shape)
print (x_test.shape)

train_model = False
model_weights_file = "CNN_ae_weights.kmdl"

# encoding_dim = 32
input_img = Input(shape=(28,28,1))

x = Conv2D(16,(3,3),activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2),padding='same')(x)
x = Conv2D(8,(3,3),activation='relu', padding='same')(x)
x = MaxPooling2D((2,2),padding='same')(x)
x = Conv2D(8,(3,3),activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2),padding='same')(x)

# from 28x28, max pooled thrice with same padding. 28-14-7-4. 7->4 is with same padding
#now invert the process
x = Conv2D(8,(3,3),activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8,(3,3),activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
#todo NOTICE there is no padding here, to match the dimensions needed.
x = Conv2D(16,(3,3),activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)


#full AE model
autoencoder = Model(input_img,decoded)
autoencoder.compile(optimizer='adadelta', loss = 'binary_crossentropy')
# autoencoder.compile(optimizer='adadelta', loss = 'mse')
#encoder model
encoder = Model (input_img,encoded)
#decoder model


if not train_model:
    autoencoder.load_weights(filepath=model_weights_file)
else:
    autoencoder.fit(x_train,x_train,epochs=50,batch_size=256,
                    shuffle=True,validation_data=(x_test,x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    autoencoder.save_weights(model_weights_file)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)


import matplotlib.pyplot as plt

n=10 #number of images to be displayed
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(True)#just for fun
    #display reconstruction
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
