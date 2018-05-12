from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
from keras.callbacks import TensorBoard
import random as rand
import copy
from keras import backend as K
from keras import metrics



train_model = False
fraction_of_data = 1.0
RewardError = False  #todo add support for negative values
Resample = True
#noisy to noisy only matters if occlude is true
InputToOutputType = 2 #1-True to True  2-True to Noisy 3-Noisy to True  4-Noisy to Noisy
                    #the other option is True input to noisy output. Noisy to Noisy makes sense only because we never truly train
                    #on weaker images, we noise them , so pay less importance. if it is True to noisy, then we are learning to
                    #occlude , which is not really our goal
Occlude = InputToOutputType != 1
Sparsity  = True
Predict_on_test = True

# dict_num_reward = {0:1,     1:1,    2:1,    3:1,    4:1,    5:1,    6:1,    7:1,    8:1,  9:1}
# dict_num_reward = {0:0,     1:0,    2:0,    3:0,    4:0,    5:0,    6:0,    7:0,    8:0,  9:0}
# dict_num_reward = {0:0,     1:0,    2:0,    3:0.3,    4:0,    5:0,    6:0.3,    7:0,    8:1,  9:0}
dict_num_reward = {0:0,     1:0.3,    2:0,    3:0,    4:0.3,    5:0,    6:0,    7:0,    8:1,  9:0}




model_weights_file_name = "weights_CNN_AE"
if RewardError: model_weights_file_name += "_RE"
if Resample: model_weights_file_name += "_R"
if Sparsity: model_weights_file_name += "_S"
model_weights_file_name += "_" + str(InputToOutputType)
model_weights_file_name += "_13_43_81" + ".kmdl"

# model_weights_file_name = "CNN_ae_weights_ResampleOcclude_NoiseToNoise_148.kmdl"

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


def keep_sample_by_reward(index, reward):
    #introduce noise into each pixel with probability determined noise_factor
    #done by first generating noise value over an array of zeros, and then
    #AVERAGING the noise with the DATA.
    cutoff= np.random.rand()
    if cutoff < reward:
        #also modify the image by adding noise based on (1-reward)
        if Occlude:
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


def mnist_reward(in_value):
    # for pure dict values
    return dict_num_reward[in_value]
    #todo add support for negative values
#=============================
if Resample:
    new_x_train_indices = [keep_sample_by_reward(i, mnist_reward(y_train[i])) for i in range(x_train.shape[0])]
    new_x_train_indices  = set(new_x_train_indices)
    try:
        new_x_train_indices.remove(-1)
    except:
        pass
else: #dont resample
    new_x_train_indices = range(x_train.shape[0])
#=============================
new_x_train_indices = list(new_x_train_indices)
new_x_train_indices = new_x_train_indices[:int(len(new_x_train_indices)/1000)*1000] #for making sure we get batchsize divisible
x_train_target = x_train[new_x_train_indices]
x_train_original = x_train_original[new_x_train_indices]
y_train = y_train[new_x_train_indices]
x_train_reward = np.array([mnist_reward(y_train[i]) for i in range(len(y_train))])
x_test_reward = np.array([mnist_reward(y_test[i]) for i in range(len(y_test))])


# encoding_dim = 32
input_img = Input(shape=(28,28,1))
if RewardError:
    input_reward = Input (shape=(1,), name="reward")

x = Conv2D(16,(3,3),activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2),padding='same')(x)
x = Conv2D(8,(3,3),activation='relu', padding='same')(x)
x = MaxPooling2D((2,2),padding='same')(x)
x = Conv2D(8,(3,3),activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2),padding='same')(x)


middle_shape = encoded.shape[1:]
middle_shape = [int(i) for i in middle_shape]
flat_layer_size = np.product(middle_shape)
flat_layer = Flatten()(encoded)
if Sparsity:
    dense_layer = Dense(flat_layer_size,activation="relu",activity_regularizer=regularizers.l1(10e-5))(flat_layer)
else:
    dense_layer = Dense(flat_layer_size, activation="relu")(flat_layer)
encoded = Reshape(middle_shape)(dense_layer)


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



#compute the reward based loss
if RewardError:
    xent_loss = K.mean(metrics.binary_crossentropy(input_img,decoded))
    reward_based_loss =  input_reward*xent_loss
    autoencoder = Model(inputs = [input_img,input_reward],outputs = [decoded])
    autoencoder.add_loss(reward_based_loss)
    autoencoder.compile(optimizer='sgd')
else:
    autoencoder = Model(input_img,decoded)
    autoencoder.compile(optimizer='adadelta', loss = 'binary_crossentropy')

#encoder model
encoder = Model (input_img,encoded)
#decoder model

if not train_model:
    autoencoder.load_weights(filepath=model_weights_file_name)
else:
    # InputToOutputType = 4  # 1-True to True  2-True to Noisy 3-Noisy to True  4-Noisy to Noisy
    # x_train_original, x_train_target
    source_images = x_train_original
    target_images = x_train_original
    if InputToOutputType == 2:
        target_images = x_train_target
    if InputToOutputType == 3 or InputToOutputType == 4:
        source_images = x_train_target

    if RewardError:
        # todo try changing the loss function to return a single value as opposed to an array and see if tensorboard works
        autoencoder.fit([source_images,x_train_reward],epochs=5,batch_size=25,
                        shuffle=True,validation_data=([x_test,x_test_reward],None))
                    # ,callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    else:
        autoencoder.fit(source_images,target_images,epochs=5,batch_size=25,
                        shuffle=True,validation_data=(x_test,x_test))

    autoencoder.save_weights(model_weights_file_name)
    print(model_weights_file_name)
#======= Done with training model,
#===== now predict and display


if Predict_on_test:
    encoded_imgs = encoder.predict(x_test)
    if RewardError:
        decoded_imgs = autoencoder.predict([x_test,x_test_reward])
    else:
        decoded_imgs = autoencoder.predict(x_test)
else:
    encoded_imgs = encoder.predict(x_train_original)
    if RewardError:
        decoded_imgs = autoencoder.predict([x_train_original,x_train_reward])
    else:
        decoded_imgs = autoencoder.predict(x_train_original)


#find the indices of two of each class
# needed_numbers = [0,1,2,3,4,5,6,7,8,9]
needed_numbers = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
# needed_numbers = [i for i in dict_num_reward.keys() if dict_num_reward[i]>0]
needed_numbers = needed_numbers * 20 #this will be more than the images, but thats ok. There are checks in place to break out
target_indices = []
curr_index = rand.randint(100,1000)


y_target = y_train
if Predict_on_test:
    y_target = y_test

for number in needed_numbers:
    while True:
        curr_index += 1
        if curr_index % len(y_target ) == 0:
            curr_index = 0
            break
        if y_target[curr_index] == number:
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

    if Predict_on_test:
        plt.imshow(x_test[target_indices[i]].reshape(28,28))
    else:
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
