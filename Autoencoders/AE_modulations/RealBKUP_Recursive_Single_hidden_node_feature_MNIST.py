from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, RepeatVector,Add
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
from keras.callbacks import TensorBoard
import random as rand
import copy
from keras import backend as K
from keras import metrics
import math
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error as mse
from skimage.transform import rescale, resize, downscale_local_mean

train_model = True
fraction_of_data = 1.0
optimizer_type = 'adadelta'
batch_size = 20
num_epochs = 5
error_function = metrics.mse
min_num_data_points = 6000
RewardError = False
RewardBasedResampling = True
#noisy to noisy only matters if occlude is true
InputToOutputType = 1 #1-True to True  2-True to Noisy 3-Noisy to True  4-Noisy to Noisy
                    #the other option is True input to noisy output. Noisy to Noisy makes sense only because we never truly train
                    #on weaker images, we noise them , so pay less importance. if it is True to noisy, then we are learning to
                    #occlude , which is not really our goal
Occlude = InputToOutputType != 1
Sparsity  = False
Array_Error = True
Invert_Img_Negative = False
Negative_Error_From_Reward = False#todo set to false as default
Predict_on_test = False
Resample = False
if RewardBasedResampling or Occlude or Invert_Img_Negative:
    Resample = True

# dict_num_reward = {0:1,     1:1,    2:1,    3:1,    4:1,    5:1,    6:1,    7:1,    8:1,  9:1}
# dict_num_reward = {0:0,     1:0,    2:0,    3:0,    4:0,    5:0,    6:0,    7:0,    8:0,  9:0}
# dict_num_reward = {0:0,     1:0,    2:0,    3:0.3,    4:0,    5:0,    6:0.3,    7:0,    8:1,  9:0}
dict_num_reward = {0:0,     1:0,    2:0.5,    3:0,    4:0,    5:0,    6:0,    7:0,    8:0,  9:0 }


def get_reward_string():
    string_repr = "_"
    for key in dict_num_reward.keys():
        if dict_num_reward[key] != 0:
            string_repr +=  str(key) + str(dict_num_reward[key]).replace(".","") + "_"
    return string_repr[:-1]
    #--end for


model_weights_file_name = "RECURSIVE_1_hidden_node_CNN_AE"
if RewardError: model_weights_file_name += "_RewErr"
if Resample: model_weights_file_name += "_Rsmpl"
if Sparsity: model_weights_file_name += "_Sprs"
if Array_Error: model_weights_file_name += "_ArrErr"
model_weights_file_name += "_" + "inOutType" + str(InputToOutputType)
model_weights_file_name += "_" + optimizer_type + "_" + str(batch_size)
model_weights_file_name += get_reward_string()

# model_weights_file_name = "CNN_ae_weights_ResampleOcclude_NoiseToNoise_148.kmdl"
#=============================================
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


#=============================================

def mnist_reward(in_value):
    # for pure dict values
    return dict_num_reward[in_value]

#=============================================
cumulative_reward = 0.0
x_test_reward = np.array([mnist_reward(y_test[i]) for i in range(len(y_test))]) #this is necessary for the test set
x_train_reward = np.array([mnist_reward(y_train[i]) for i in range(len(y_train))]) #this is necessary for the train set
x_train_target = x_train_original
test_fitted_length = int(x_test.shape[0] / batch_size) * batch_size
x_test = x_test[0:test_fitted_length]
x_test_reward = x_test_reward[0:test_fitted_length]

def keep_sample_by_reward():

    global x_train_target,x_train_original,x_train_reward,y_train
    sampled_xtrain_target = []
    sampled_ytrain = []
    sampled_x_train_original = []
    sampled_x_train_reward = []
    while len(sampled_x_train_reward) < min_num_data_points:
        index_list = np.array(list(range(x_train.shape[0])))
        np.random.shuffle(index_list)
        for index in index_list:
            curr_reward = mnist_reward(y_train[index])
            if curr_reward == 0 and RewardError == True:
                continue #no point in a data point with zero reward, as the error would be 0
            # formula for (sampling) = e ^ (-1 * sum * reward) / e ^ (| sum |)
            prob_sampling = curr_reward
            # prob_sampling
            cutoff = np.random.rand()
            if cutoff < prob_sampling or RewardBasedResampling == False:
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
                sampled_ytrain.append(y_train[index])
                sampled_x_train_reward.append(curr_reward)
                sampled_x_train_original.append(x_train_original[index])
        #---END for loop through index of data points
    #-end while loop
    #now trim to the batch size
    fitted_length = int(len(sampled_x_train_reward)/batch_size)*batch_size
    x_train_target = np.array(sampled_xtrain_target[0:fitted_length])
    x_train_original = np.array(sampled_x_train_original[0:fitted_length])
    x_train_reward = np.array(sampled_x_train_reward[0:fitted_length])
    y_train = np.array(sampled_ytrain[0:fitted_length])


#=============================

if Resample and train_model:
    keep_sample_by_reward()
else: #dont resample
    x_train_target = x_train_original
    pass
#=============================


x_test_approx = np.zeros(x_test.shape)
x_train_approx = np.zeros(x_train_original.shape)

# encoding_dim = 32
input_img = Input(shape=(28,28,1))
prev_approx_img = Input(shape=(28,28,1))
target_img = Input(shape=(28,28,1))

"""
NN design algo
There are many repeat NN, each trained sequentially.
1) Train v1, evaluate the results for the dataset.
2) subtract the original dataset with the evaluated results. 
3) Save v1 weights with _v1 suffix
4) Reset the weights (? or not) and train the model on the modified dataset.
5) Repeat

Debug by looking at the resultant images wrt to the original or prev data set.
Debug by only training on 8s to begin with.
"""


x = Conv2D(8,(2,2),activation='tanh')(input_img)

# x = MaxPooling2D((2,2),padding='same')(x)
# x = Conv2D(8,(2,2),activation='tanh')(x)
# x = MaxPooling2D((2,2),padding='same')(x)
# x = Conv2D(8,(2,2),activation='tanh')(x)

flat_layer = Flatten()(x)
dense_layer = Dense(1, activation="tanh")(flat_layer)

dense_feature_layer = Dense(int(28*28),activation="tanh")(dense_layer)
encoded = Reshape([28,28,1])(dense_feature_layer)
x = Conv2D(8,(2,2),activation='tanh',padding="same")(encoded)
decoded = Conv2D(1,(2,2),activation='tanh',padding='same')(x)
decoded = Add()([decoded,prev_approx_img])


if Array_Error:
    loss_value = error_function(target_img, decoded)
else:  # not array error
    loss_value = K.mean(error_function(target_img, decoded))

autoencoder = Model([input_img,prev_approx_img,target_img], decoded)
autoencoder.add_loss(loss_value)
autoencoder.compile(optimizer=optimizer_type)


#
# autoencoder = Model([input_img,target_img],decoded)
# # autoencoder.add_loss(xent_loss)
# autoencoder.compile(optimizer=optimizer_type,loss="mse")

#encoder model
encoder = Model (input_img,encoded)
#decoder model


#find the indices of two of each class
# needed_numbers = [0,1,2,3,4,5,6,7,8,9]
# needed_numbers = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
# needed_numbers =   [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]
# needed_numbers = [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
# needed_numbers = [7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]
# needed_numbers =   [4,4,4,4,4,4,4,4,4,1,1,1,1,1,8,8,8,8,8,8]
needed_numbers = [i for i in dict_num_reward.keys() if dict_num_reward[i]>0]
needed_numbers = needed_numbers * 20 #this will be more than the images, but thats ok. There are checks in place to break out
target_indices = []
curr_index = rand.randint(100,1000)


y_target = y_train
source_images = x_train_target
if Predict_on_test:
    y_target = y_test
    source_images = x_test
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


if not train_model:
    pass#todo fill the iterative load and evaluate here as well. OR BETTER create many copies of the model and load
    # autoencoder.load_weights(filepath=model_weights_file_name)
else:
    # InputToOutputType = 4  # 1-True to True  2-True to Noisy 3-Noisy to True  4-Noisy to Noisy
    # x_train_original, x_train_target
    source_images = x_train_original
    target_images = x_train_original
    if InputToOutputType == 2:
        target_images = x_train_target
    if InputToOutputType == 3 or InputToOutputType == 4:
        source_images = x_train_target

    #todo THINK IF THIS IS GOOD. I think it is, so all the AE units start with the same weights (connected)
    autoencoder.save_weights("default_init_weights.kmdl")
    output_images_iter_list = []
    output_images_testSet_iter_list = []

    if train_model == False:
        for i in range(3):
            #todo complete this code. It should not be here, only when predicting
            autoencoder.load_weights(model_weights_file_name + "_L" + str(i) + ".kmdl")


    else:
        for i in range(3):
            autoencoder.load_weights("default_init_weights.kmdl")
            if RewardError:
                autoencoder.fit([source_images,target_images,x_train_reward],epochs=num_epochs ,batch_size=batch_size,
                                shuffle=True,validation_data=([x_test,x_test,x_test_reward],None))
                        # ,callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
            else:
                autoencoder.fit([source_images,x_train_approx,target_images],epochs=num_epochs ,batch_size=batch_size,
                                shuffle=True,validation_data=([x_test,x_test_approx,x_test],None))
            autoencoder.save_weights(model_weights_file_name+"_L"+str(i)+".kmdl")
            #now update the data for the next iteration.
            output_images = autoencoder.predict([source_images,x_train_approx,target_images])
            output_images_iter_list.append(output_images)
            output_images_testSet = autoencoder.predict([x_test,x_test_approx,x_test])
            output_images_testSet_iter_list.append(output_images_testSet)

            n = 20  # number of images to be displayed
            plt.figure(figsize=(n,4))
            plt.suptitle(model_weights_file_name)
            for i in range(n):
                if i >= len(target_indices):
                    break
                ax = plt.subplot(2, n, i + 1)

                plt.imshow(source_images[target_indices[i]].reshape(x_train.shape[1], x_train.shape[2]))

                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(True)  # just for fun
                # display reconstruction
                ax = plt.subplot(2, n, i + 1 + n)
                plt.imshow(output_images[target_indices[i]].reshape(x_train.shape[1], x_train.shape[2]))
                # a = source_images[target_indices[i]].reshape(source_images[0].shape[:-1])
                # b = decoded_imgs[target_indices[i]].reshape(source_images[0].shape[:-1])
                # final_mse = mse(a,b)
                # plt.title("mse=", str(final_mse))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()


            x_train_approx = output_images # todo NOTE: do NOT compound the images, else the error of the first image adds on.
            # source_images = source_images #YES keep the source and target images the same
            x_test_approx = output_images_testSet

            # n = 20  # number of images to be displayed
            # plt.figure(figsize=(20, 4))
            # plt.suptitle(model_weights_file_name)
            # for i in range(n):
            #     if i >= len(target_indices):
            #         break
            #     ax = plt.subplot(2, n, i + 1)
            #
            #     plt.imshow(source_images[target_indices[i]].reshape(x_train.shape[1], x_train.shape[2]))
            #
            #     plt.gray()
            #     ax.get_xaxis().set_visible(False)
            #     ax.get_yaxis().set_visible(True)  # just for fun
            #     # display reconstruction
            #     ax = plt.subplot(2, n, i + 1 + n)
            #     plt.imshow(output_images[target_indices[i]].reshape(x_train.shape[1], x_train.shape[2]))
            #     # a = source_images[target_indices[i]].reshape(source_images[0].shape[:-1])
            #     # b = decoded_imgs[target_indices[i]].reshape(source_images[0].shape[:-1])
            #     # final_mse = mse(a,b)
            #     # plt.title("mse=", str(final_mse))
            #     plt.gray()
            #     ax.get_xaxis().set_visible(False)
            #     ax.get_yaxis().set_visible(False)
            # plt.show()


decoded_imgs = output_images_iter_list[0]
for delta_set in output_images_iter_list[1:]:
    decoded_imgs += delta_set


n=20 #number of images to be displayed
plt.figure(figsize=(20,4))
plt.suptitle(model_weights_file_name)

for i in range(n):
    if i >= len(target_indices):
        break
    ax = plt.subplot(2,n,i+1)


    plt.imshow(source_images[target_indices[i]].reshape(x_train.shape[1], x_train.shape[2]))

    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(True)#just for fun
    #display reconstruction
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[target_indices[i]].reshape(x_train.shape[1], x_train.shape[2]))
    # a = source_images[target_indices[i]].reshape(source_images[0].shape[:-1])
    # b = decoded_imgs[target_indices[i]].reshape(source_images[0].shape[:-1])
    # final_mse = mse(a,b)
    # plt.title("mse=", str(final_mse))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
