from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
from keras.callbacks import TensorBoard


train_model = False
model_weights_file = "Denoising_CNN_ae_weights.kmdl"

#prep the data
(x_train,_) , (x_test,_) = mnist.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = x_train.reshape((len(x_train),28,28,1))
x_test = x_test.reshape((len(x_test),28,28,1))

noise_factor = 0.5
x_train_noisy = x_train + noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_train.shape)
x_test_noisy = x_test + noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

# THIS is to see the noisy data
# import matplotlib.pyplot as plt
# n=10 #number of images to be displayed
# plt.figure(figsize=(20,2))
# for i in range(n):
#     ax = plt.subplot(1,n,i+1)
#     plt.imshow(x_test_noisy[i].reshape(28,28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(True)#just for fun
#     #display reconstruction
# plt.show()


input_img = Input(shape=(28,28,1))
#todo NOTE that we have added more filters than the previous CNN AE
#AS WELL as less compression
x = Conv2D(32,(3,3),activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2),padding='same')(x)
x = Conv2D(32,(3,3),activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2),padding='same')(x)
#now the representation is 7,7,32
#todo NOTE THAT this is MORE than the input dimensions. Denoising needs that .


x = Conv2D(32,(3,3),activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(32,(3,3),activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
#todo NOTICE there is no padding here, to match the dimensions needed.
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
    autoencoder.fit(x_train_noisy,x_train,epochs=100,batch_size=256,
                    shuffle=True,validation_data=(x_test_noisy,x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/denoising_ae',
                                histogram_freq=0,write_graph=False)])
    autoencoder.save_weights(model_weights_file)

encoded_imgs = encoder.predict(x_test_noisy)
decoded_imgs = autoencoder.predict(x_test_noisy)


import matplotlib.pyplot as plt

n=10 #number of images to be displayed
plt.figure(figsize=(20,6))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test_noisy[i+5].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(True)#just for fun
    #display reconstruction
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i+5].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
