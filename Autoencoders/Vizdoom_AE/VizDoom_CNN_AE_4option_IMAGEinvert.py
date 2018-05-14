from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, RepeatVector
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
from keras.callbacks import TensorBoard
import random as rand
import copy
from keras import backend as K
from keras import metrics
import pickle
from sklearn import preprocessing

data_source_file_name = "vizdoom_memory_52_52.p"

train_model = True
fraction_of_data = 1.0
optimizer_type = 'adadelta'
batch_size = 25
num_epochs = 5
RewardError = False
#todo add support for negative reward values
Resample = False
#noisy to noisy only matters if occlude is true
InputToOutputType = 1 #1-True to True  2-True to Noisy 3-Noisy to True  4-Noisy to Noisy
                    #the other option is True input to noisy output. Noisy to Noisy makes sense only because we never truly train
                    #on weaker images, we noise them , so pay less importance. if it is True to noisy, then we are learning to
                    #occlude , which is not really our goal
Occlude = InputToOutputType != 1
Sparsity  = False
Array_Error = False
Invert_Img_Negative = False
Negative_Error_From_Reward = False #todo set to false as default
Predict_on_test = True


"""
CHANGES MADE
InputToOutputType = 1
Resample = False

"""


model_weights_file_name = "weights_CNN_AE"
if RewardError: model_weights_file_name += "_RewErr"
if Resample: model_weights_file_name += "_Rsmpl"
if Sparsity: model_weights_file_name += "_Sprs"
if Array_Error: model_weights_file_name += "_ArrErr"
model_weights_file_name += "_" + "inOutType" + str(InputToOutputType)
model_weights_file_name += "_" + optimizer_type + "_" + str(batch_size)
model_weights_file_name += ".kmdl"

# model_weights_file_name = "CNN_ae_weights_ResampleOcclude_NoiseToNoise_148.kmdl"
#=============================================
#prep the data

s1_images = None
s2_images = None
reward = None
with open(data_source_file_name,"rb") as data_source:
    s1_images = pickle.load(data_source)
    s2_images = pickle.load(data_source)
    reward = pickle.load(data_source)

s1_images = s1_images.reshape(list(s1_images.shape)+[1])
s2_images = s2_images.reshape(list(s2_images.shape)+[1])
max_reward_value = np.max(np.abs(reward))
reward = reward/max_reward_value
x_train = s1_images
x_train_original = copy.deepcopy(x_train)
x_test = x_train_original
x_train_reward = reward
x_test_reward = reward

#=============================================

def keep_sample_by_reward(index, reward):
    #introduce noise into each pixel with probability determined noise_factor
    #done by first generating noise value over an array of zeros, and then
    #AVERAGING the noise with the DATA.

    cutoff = np.random.rand()
    if cutoff < abs(reward):

        main_image = x_train[index]
        #if image is negative, invert the image
        if reward < 0 and Invert_Img_Negative:
            main_image = 1-main_image
            main_image = np.abs(main_image)

        #also modify the image by adding noise based on (1-reward)
        if Occlude:
            noise_mask = np.random.rand(x_train.shape[1], x_train.shape[2], x_train.shape[3])
            noise_mask = np.less(noise_mask,1-abs(reward))  # so if the noise factor was 0.4 (reward = 0.6), then
            # only those nodes where value is less than 0.4 will be 1
            noise_layer = np.random.rand(x_train.shape[1], x_train.shape[2], x_train.shape[3])# THIS is the actual noise value. DIFFERENT from the one used to generate mask
            # noise_layer = np.zeros(shape=(28, 28, 1)) #THIS is if you want the background to go to black.
            x_train[index] = main_image*(1 - noise_mask) + noise_layer * noise_mask
        return index
    else:
        return -1


#=============================
if Resample:
    new_x_train_indices = [keep_sample_by_reward(i, reward[i]) for i in range(x_train.shape[0])]
    new_x_train_indices  = set(new_x_train_indices)
    try:
        new_x_train_indices.remove(-1)
    except:
        pass
else: #dont resample
    new_x_train_indices = range(x_train.shape[0])
#=============================
new_x_train_indices = list(new_x_train_indices)
new_x_train_indices = new_x_train_indices[:int(len(new_x_train_indices)/batch_size)*batch_size] #for making sure we get batchsize divisible
x_train_target = x_train[new_x_train_indices]
x_train_original = x_train_original[new_x_train_indices]
x_train_reward = np.array([reward[i] for i in new_x_train_indices])


if not Negative_Error_From_Reward:
    x_train_reward = np.abs(x_train_reward)
    x_test_reward = np.abs(x_test_reward)

# encoding_dim = 32
input_img = Input(shape=x_train.shape[1:])
target_img = Input(shape=x_train_target.shape[1:])
if RewardError:
    input_reward = Input(shape=(1,), name="reward")
    if Array_Error:
        #repeat and reshape to match the image
        target_img_shape = target_img.shape[1:-1]
        target_img_shape = [int(i) for i in target_img_shape]
        target_img_num_dims = np.prod(np.array(target_img_shape))
        input_reward_repeated = RepeatVector(target_img_num_dims)(input_reward)
        input_reward_reshaped = Reshape(target_img_shape)(input_reward_repeated)


x = Conv2D(16,(3,3),activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2),padding='same')(x)
x = Conv2D(8,(3,3),activation='relu', padding='same')(x)
x = MaxPooling2D((2,2),padding='same')(x)
x = Conv2D(8,(3,3),activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2),padding='same')(x)
#NOW we are in the middle
middle_shape = encoded.shape[1:]
middle_shape = [int(i) for i in middle_shape]
flat_layer_size = np.product(middle_shape)
flat_layer = Flatten()(encoded)
if Sparsity:
    dense_layer = Dense(flat_layer_size,activation="relu",activity_regularizer=regularizers.l1(10e-5))(flat_layer)
else:
    dense_layer = Dense(flat_layer_size, activation="relu")(flat_layer)
encoded = Reshape(middle_shape)(dense_layer)
#NOW we are in the end of the AE
# from 28x28, max pooled thrice with same padding. 28-14-7-4. 7->4 is with same padding
#now invert the process
x = Conv2D(8,(3,3),activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8,(3,3),activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
#todo NOTICE there is no padding here, to match the dimensions needed.
x = Conv2D(16,(3,3),activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1,(3,3),activation='relu',padding='same')(x)



#compute the reward based loss

if RewardError:
    if Array_Error:
        xent_loss = metrics.binary_crossentropy(target_img, decoded)
        reward_based_loss = input_reward_reshaped* xent_loss
    else:
        xent_loss = K.mean(metrics.binary_crossentropy(target_img, decoded))
        reward_based_loss = input_reward * xent_loss

    autoencoder = Model(inputs=[target_img, input_img, input_reward], outputs=[decoded])
    autoencoder.add_loss(reward_based_loss)
    autoencoder.compile(optimizer=optimizer_type)
else:#not reward error
    if Array_Error:
        xent_loss = metrics.binary_crossentropy(target_img, decoded)
    else:#not array error
        xent_loss = K.mean(metrics.binary_crossentropy(target_img, decoded))

    autoencoder = Model([target_img,input_img], decoded)
    autoencoder.add_loss(xent_loss)
    autoencoder.compile(optimizer=optimizer_type)

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
        autoencoder.fit([target_images,source_images,x_train_reward],epochs=num_epochs,batch_size=batch_size,
                        shuffle=True,validation_data=([x_test,x_test,x_test_reward],None))
                # ,callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    else:
        autoencoder.fit([target_images,source_images],epochs=num_epochs,batch_size=batch_size,
                        shuffle=True,validation_data=([x_test,x_test],None))

    autoencoder.save_weights(model_weights_file_name)
    print(model_weights_file_name)
#======= Done with training model,
#===== now predict and display


if Predict_on_test:
    encoded_imgs = encoder.predict(x_test)
    if RewardError:
        decoded_imgs = autoencoder.predict([x_test,x_test,x_test_reward])
    else:
        decoded_imgs = autoencoder.predict([x_test,x_test])
else:
    encoded_imgs = encoder.predict(x_train_original)
    if RewardError:
        decoded_imgs = autoencoder.predict([x_train_original,x_train_original,x_train_reward])
    else:
        decoded_imgs = autoencoder.predict([x_train_original,x_train_original])



target_indices = []
curr_index = rand.randint(100,1000)
target_indices = range(curr_index,curr_index+10)



import matplotlib.pyplot as plt
n=20 #number of images to be displayed
plt.figure(figsize=(20,4))
plt.suptitle(model_weights_file_name)

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
