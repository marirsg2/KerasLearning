from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np

#prep the data
(x_train,_) , (x_test,_) = mnist.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)

train_model = True
model_weights_file = "Deep_dense_ae_weights.kmdl"

encoding_dim = 32
input_img = Input(shape=(784,))

encoded = Dense(128, activation = 'relu',)(input_img)
encoded = Dense(64, activation = 'relu',)(encoded)
encoded = Dense(encoding_dim, activation = 'relu',)(encoded)
decoded = Dense(64, activation = 'relu',)(encoded)
decoded = Dense(128, activation = 'relu',)(decoded)
decoded = Dense (784,activation = 'sigmoid')(decoded) #sigmoid because zero or 1
# decoded = Dense (784,activation = 'relu')(decoded) #sigmoid because zero or 1
# decoded = Dense (784,activation = 'linear')(decoded) #sigmoid because zero or 1

#full AE model
autoencoder = Model(input_img,decoded)
autoencoder.compile(optimizer='adadelta', loss = 'binary_crossentropy')
# autoencoder.compile(optimizer='adadelta', loss = 'mse')
#encoder model
encoder = Model (input_img,encoded)

#decoder model
encoded_input = Input(shape= (encoding_dim,))
decoder_layer = (autoencoder.layers[-3])(encoded_input)
decoder_layer = (autoencoder.layers[-2])(decoder_layer)
decoder_layer = (autoencoder.layers[-1])(decoder_layer)
decoder = Model(encoded_input, decoder_layer)


if not train_model:
    autoencoder.load_weights(filepath=model_weights_file)
else:
    autoencoder.fit(x_train,x_train,epochs=50,batch_size=256,
                    shuffle=True,validation_data=(x_test,x_test))
    autoencoder.save_weights(model_weights_file)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

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
