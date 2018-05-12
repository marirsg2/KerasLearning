

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

train_weights = True
model_weights_file = "vae_weights_2.kmdl"

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

x = Input(batch_shape=(batch_size,original_dim))
h = Dense (intermediate_dim,activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

#--------------------------------------
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0],latent_dim),\
                              mean = 0, stddev=epsilon_std)
    #todo notice below that /2 of log variance is sqrt of variance = std_dev
    return z_mean + K.exp(z_log_var/2) * epsilon
#--------------------------------------

# note that "output_shape" isn't necessary with the TensorFlow backend
#todo NOTE that the output shape is already determined by z_mean and log_Var (of size latent dim)
#since Lambda layer is merely applying a matrix function, the output shape CAN be assumed to be that
#of the inputs(I think). If different, then need to specify.
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

#we want some layers to be reused, and so will be separately instantiated
#notice that they are not given the additional argument of the previous layer
decoder_h = Dense(intermediate_dim, activation='relu')#h = hidden
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)#NOW we connect it to the "z" (latent var) layer that sampled from N(0,1) and mean,var
x_decoded_mean = decoder_mean(h_decoded)
#instantiate VAE model
vae = Model(x,x_decoded_mean)
#todo read carefully
#computing the VAE loss
xent_loss = original_dim*metrics.binary_crossentropy(x,x_decoded_mean)
#this is the kl divergence between ? two normal distributions. Easy in closed form,
#and one of the reasons we sample from another normal distribution. Using a parametric distribution
#like normal, can help by getting such closed forms.
kl_loss =  -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
#todo UNSURE if this model structure is the best. I think I've seen it where the KL
#divergence optimizes part of the model, and the reconstruction error alone is used
#to update the full model. So KL loss does not affect the decoder.
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

#now train VAE on MNIST digits
# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

if train_weights:
    vae.fit(x_train,shuffle=True,epochs = epochs, batch_size=batch_size,\
            validation_data=(x_test,None))
    vae.save_weights(model_weights_file)
else:
    vae.load_weights(model_weights_file)

#build a model to project inputs on the latent space
#we only need the mean to plot in 2-d
encoder = Model(x,z_mean)
x_test_encoded = encoder.predict(x_test,batch_size=batch_size)
plt.figure(figsize=(6,6))
#recall that we only had TWO latent dimensions
plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c = y_test)
plt.colorbar()
plt.show()

#we can also build digits with a generator that sample from the learned distribution parameters
decoder_input = Input(shape = (latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

#display 2D manifold of the digits
n=15
digit_size = 28
figure = np.zeros((digit_size*n,digit_size*n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
#now put it into a single figure (image)
for i,yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi,yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size,digit_size)
        #"figure" is filled with zeros. Fill one image's worth of pixels with the values in "digit"
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10,10))
plt.imshow(figure,cmap='Greys_r')
plt.show()