from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, RepeatVector
from keras.models import Model
from keras import optimizers
from keras.datasets import mnist
from keras import regularizers
import numpy as np
from keras.callbacks import TensorBoard
import random as rand
import copy
from keras import backend as K
from keras import metrics
import pickle
from sklearn.metrics import mean_squared_error as mse
import math

data_source_file_name = "vizdoom_memory_100_100.p"

train_model = True
fraction_of_data = 1.0
optimizer_type = 'sgd'
learning_rate = 0.0001
batch_size = 1
num_epochs = 5
min_num_data_points = 1000
error_function = metrics.mean_squared_error
RewardError = True
RewardBasedResampling = True
#noisy to noisy only matters if occlude is true
InputToOutputType = 1 #1-True to True  2-True to Noisy 3-Noisy to True  4-Noisy to Noisy
                    #the other option is True input to noisy output. Noisy to Noisy makes sense only because we never truly train
                    #on weaker images, we noise them , so pay less importance. if it is True to noisy, then we are learning to
                    #occlude , which is not really our goal
Occlude = InputToOutputType != 1
Sparsity  = False
Array_Error = True
Negative_Error_From_Reward = True#todo set to false as default
Predict_on_test = False
Resample = False
if RewardBasedResampling or Occlude :
    Resample = True


"""
CHANGES MADE
InputToOutputType = 1
Resample = False

"""


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

#REMOVE ALL IMAGES where reward == 0, ONLY BECAUSE In THIS DATA SET
# REWARD = 0 means a black image in this data set
non_zero_indices = np.where(reward != 0)[0]
s1_images = s1_images[non_zero_indices]
s2_images = s2_images[non_zero_indices]
reward = reward[non_zero_indices]

s1_images = s1_images.reshape(list(s1_images.shape)+[1])
s2_images = s2_images.reshape(list(s2_images.shape)+[1])
max_reward_value = np.max(np.abs(reward))
reward = reward/max_reward_value
x_train = s1_images
x_train_original = copy.deepcopy(x_train)
x_train_untouched = copy.deepcopy(x_train)
x_test = x_train_original
x_train_reward = reward
x_test_reward = reward



#=====================================================
model_weights_file_name = "weights_CNN_AE"
if RewardError: model_weights_file_name += "_RewErr"
if Resample: model_weights_file_name += "_Rsmpl"
if Sparsity: model_weights_file_name += "_Sprs"
if Array_Error: model_weights_file_name += "_ArrErr"
model_weights_file_name += "_" + "inOutType" + str(InputToOutputType)
model_weights_file_name += "_" + optimizer_type + "_" + str(learning_rate) +"_" + str(batch_size)
model_weights_file_name += "_" +  str(x_train.shape[1])+ "by" + str(x_train.shape[2])
model_weights_file_name += ".kmdl"

#=============================================

cumulative_reward = 0.0
def keep_sample_by_reward():

    global cumulative_reward,x_train_target,x_train_original,x_train_reward,reward
    sampled_xtrain_target = []
    sampled_x_train_original = []
    sampled_x_train_reward = []
    while len(sampled_x_train_reward) < min_num_data_points:
        index_list = np.array(list(range(x_train.shape[0])))
        np.random.shuffle(index_list)
        for index in index_list:
            curr_reward = reward[index]
            # formula for (sampling) = e ^ (-1 * sum * reward) / e ^ (| sum |)
            prob_sampling = math.exp(-1*cumulative_reward*curr_reward)/math.exp(abs(cumulative_reward))
            # prob_sampling
            cutoff = np.random.rand()
            if cutoff < prob_sampling or RewardBasedResampling == False:
                print(curr_reward, " ", cumulative_reward, " ",index)
                cumulative_reward += curr_reward#ONLY UPDATE if it was successfully sampled
                main_image = copy.deepcopy(x_train[index])

                #also modify the image by adding noise based on (1-reward)
                if Occlude:
                    noise_mask = np.random.rand(x_train.shape[1], x_train.shape[2], x_train.shape[3])
                    noise_mask = np.less(noise_mask,1-abs(curr_reward))  # so if the noise factor was 0.4 (reward = 0.6), then
                    # only those nodes where value is less than 0.4 will be 1
                    noise_layer = np.random.rand(x_train.shape[1], x_train.shape[2], x_train.shape[3])# THIS is the actual noise value. DIFFERENT from the one used to generate mask
                    # noise_layer = np.zeros(shape=(28, 28, 1)) #THIS is if you want the background to go to black.
                    main_image = main_image*(1 - noise_mask) + noise_layer * noise_mask
                #save this
                sampled_xtrain_target.append(main_image)
                sampled_x_train_reward.append(curr_reward)
                sampled_x_train_original.append(x_train_original[index])
        #---END for loop through index of data points
    #-end while loop
    #now trim to the batch size
    fitted_length = int(len(sampled_x_train_reward)/batch_size)*batch_size
    x_train_target = np.array(sampled_xtrain_target[0:fitted_length])
    x_train_original = np.array(sampled_x_train_original[0:fitted_length])
    x_train_reward = np.array(sampled_x_train_reward[0:fitted_length])

#=============================
index_array = np.array(list(range(x_train.shape[0])))
if Resample and train_model:
    keep_sample_by_reward()
else: #dont resample
    pass
#=============================


print(" count positive reward samples = ", np.where(x_train_reward>0)[0].shape)
print(" count negative reward samples = ", np.where(x_train_reward<0)[0].shape)
print("average reward =", np.average(x_train_reward))


#===========================================================
#===========================================================
#
# target_indices = []
# curr_index = rand.randint(10,100)
# target_indices = range(curr_index,2*(curr_index+10))
#
# import matplotlib.pyplot as plt
# n=20 #number of images to be displayed
# plt.figure(figsize=(20,4))
#
# for i in range(n):
#     if i >= len(target_indices):
#         break
#     ax = plt.subplot(2,n,i+1)
#     if Predict_on_test:
#         source_images = x_test
#         source_reward = x_test_reward
#     else:
#         source_images = x_train_target
#         source_reward = x_train_reward
#     plt.imshow(source_images[target_indices[i]].reshape(x_train.shape[1], x_train.shape[2]))
#     plt.title(str(source_reward[target_indices[i]]))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(True)#just for fun
#     #display reconstruction
#     ax = plt.subplot(2,n,i+1+n)
#     plt.imshow(source_images[target_indices[2*i]].reshape(x_train.shape[1], x_train.shape[2]))
#     # final_mse = mse(a,b)#TODO PRINT MSE
#     plt.title(str(source_reward[target_indices[2*i]]))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

#===========================================================
#===========================================================



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
        original_loss = error_function(target_img, decoded)
        reward_based_loss = input_reward_reshaped * original_loss

        #todo this is not working. Need K.ones() which is not there for tf. Find answer if needed
        # target_shape = input_reward_reshaped.shape[1:-1]
        # target_shape = [int(i) for i in target_shape]
        # reward_based_loss = np.ones(shape = input_reward_reshaped.shape) * original_loss
    else:
        original_loss = K.mean(error_function(target_img, decoded))
        reward_based_loss = input_reward * original_loss

    autoencoder = Model(inputs=[target_img, input_img, input_reward], outputs=[decoded])
    autoencoder.add_loss(reward_based_loss)

else:#not reward error
    if Array_Error:
        original_loss = error_function(target_img, decoded)
    else:#not array error
        original_loss = K.mean(error_function(target_img, decoded))

    autoencoder = Model([target_img,input_img], decoded)
    autoencoder.add_loss(original_loss)

optimizer_instance = optimizers.Adadelta()
if optimizer_type == "sgd":
    optimizer_instance = optimizers.SGD(lr=learning_rate)

autoencoder.compile(optimizer=optimizer_instance)

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
    encoded_imgs = encoder.predict(x_train_target)
    if RewardError:
        decoded_imgs = autoencoder.predict([x_train_target,x_train_target,x_train_reward])
    else:
        decoded_imgs = autoencoder.predict([x_train_target,x_train_target])


n=20 #number of images to be displayed
if Predict_on_test:
    source_images = x_test
    source_reward = x_test_reward
else:
    source_images = x_train_target
    source_reward = x_train_reward

target_indices = [i for i in range(source_reward.shape[0]) if source_reward[i] > 0]
# target_indices = []
# curr_index = rand.randint(0,100)
# target_indices = range(curr_index,curr_index+n)


import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
plt.suptitle(model_weights_file_name)

for i in range(n):
    if i >= len(target_indices):
        break
    ax = plt.subplot(2,n,i+1)
    plt.imshow(source_images[target_indices[i]].reshape(x_train.shape[1], x_train.shape[2]))
    plt.title(str(source_reward[target_indices[i]]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(True)#just for fun
    #display reconstruction
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[target_indices[i]].reshape(x_train.shape[1], x_train.shape[2]))
    plt.title(str(source_reward[target_indices[i]]))
    # a = source_images[target_indices[i]].reshape(source_images[0].shape[:-1])
    # b = decoded_imgs[target_indices[i]].reshape(source_images[0].shape[:-1])
    # final_mse = mse(a,b)
    # plt.title("mse=")#, str(final_mse))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
