from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
from keras.callbacks import TensorBoard
import random as rand
import copy

fraction_of_data = 1.0
#prep the data
(x_train,y_train) , (x_test,y_test) = mnist.load_data()
x_train = x_train[0:int(len(x_train)*fraction_of_data)]
y_train = y_train[0:int(len(y_train)*fraction_of_data)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = x_train.reshape((len(x_train),28,28,1))
x_test = x_test.reshape((len(x_test),28,28,1))
x_train_original = copy.deepcopy(x_train)
print (x_train.shape)
print (x_test.shape)

train_model = False
model_weights_file = "CNN_ae_weights_ResampleOcclude_NoiseToNoise_148.kmdl"

def keep_sample_by_reward(index, reward):
    #introduce noise into each pixel with probability determined noise_factor
    #done by first generating noise value over an array of zeros, and then
    #AVERAGING the noise with the DATA.
    cutoff= np.random.rand()
    if cutoff < reward:
        #also modify the image by adding noise based on (1-reward)
        noise_mask = np.random.rand(28, 28, 1)
        noise_mask = np.less(noise_mask,1-reward)  # so if the noise factor was 0.4 (reward = 0.6), then
        # only those nodes where value is less than 0.4 will be 1
        noise_layer = np.random.rand(28, 28,1)  # THIS is the actual noise value. DIFFERENT from the one used to generate mask
        # noise_layer = np.zeros(shape=(28, 28, 1)) #THIS is if you want the background to go to black.
        x_train[index] = x_train[index]*(1 - noise_mask) + noise_layer * noise_mask
        return index
    else:
        return -1

#=============================

dict_num_reward = {0:1,     1:1,    2:1,    3:1,    4:1,    5:1,    6:1,    7:1,    8:1,  9:1}
# dict_num_reward = {0:0,     1:0,    2:0,    3:0,    4:0,    5:0,    6:0,    7:0,    8:0,  9:0}
# dict_num_reward = {0:0,     1:0,    2:0,    3:0.3,    4:0,    5:0,    6:0.3,    7:0,    8:1,  9:0}
# negative_ratio = 0.2
# negative_proportion = 1
def mnist_reward(in_value):
    # for pure dict values
    return dict_num_reward[in_value]
    #for random negative values
    # value = dict_num_reward[in_value]
    # rand_val = rand.uniform(0,1)
    # if rand_val < negative_ratio:
    #     return value*-1*negative_proportion
    # else:
    #     return value


# 1-mnist_reward = noise factor, and tells us how much noise to generate
new_x_train_indices = [keep_sample_by_reward(i, mnist_reward(y_train[i])) for i in range(x_train.shape[0])]
new_x_train_indices  = set(new_x_train_indices)
try:
    new_x_train_indices.remove(-1)
except:
    pass
new_x_train_indices = list(new_x_train_indices)
new_x_train_indices = new_x_train_indices[:int(len(new_x_train_indices)/1000)*1000]
x_train_target = x_train[new_x_train_indices]
x_train_original = x_train_original[new_x_train_indices]
y_train = y_train[new_x_train_indices]

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
    autoencoder.fit(x_train_original,x_train_target,epochs=25,batch_size=25,
                    shuffle=True,validation_data=(x_test,x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    autoencoder.save_weights(model_weights_file)
    print(model_weights_file)

encoded_imgs = encoder.predict(x_train_original)
decoded_imgs = autoencoder.predict(x_train_original)


#find the indices of two of each class
# needed_numbers = [0,1,2,3,4,5,6,7,8,9]
# needed_numbers = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
needed_numbers = [i for i in dict_num_reward.keys() if dict_num_reward[i]>0]
needed_numbers = needed_numbers * 20 #this will be more than the images, but thats ok. There are checks in place to break out

target_indices = []
curr_index = rand.randint(100,1000)
for number in needed_numbers:
    while True:
        curr_index += 1
        if curr_index % len(y_train) == 0:
            curr_index = 0
            break
        if y_train[curr_index] == number:
            target_indices.append(curr_index)
            break
    #end while
#end outer for


import matplotlib.pyplot as plt
n=20 #number of images to be displayed
plt.figure(figsize=(20,4))
for i in range(n):
    if i >= len(target_indices):
        break
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_train_original[target_indices[i]].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(True)#just for fun
    #display reconstruction
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[target_indices[i]].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
