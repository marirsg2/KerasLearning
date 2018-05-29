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
import math
import pickle
from sklearn.metrics import mean_squared_error as mse


train_model = False
fraction_of_data = 1.0
optimizer_type = 'adadelta'
batch_size = 5
num_epochs = 10
error_function = metrics.binary_crossentropy
min_num_data_points = 3000
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
Invert_Img_Negative = False
Negative_Error_From_Reward = True #todo set to false as default
Predict_on_test = True
Resample = False
if RewardBasedResampling or Occlude or Invert_Img_Negative:
    Resample = True

# dict_num_reward = {0:1,     1:1,    2:1,    3:1,    4:1,    5:1,    6:1,    7:1,    8:1,  9:1}
# dict_num_reward = {0:0,     1:0,    2:0,    3:0,    4:0,    5:0,    6:0,    7:0,    8:0,  9:0}
# dict_num_reward = {0:0,     1:0,    2:0,    3:0.3,    4:0,    5:0,    6:0.3,    7:0,    8:1,  9:0}
dict_num_reward = {0:0,     1:-0.3,    2:0,    3:0,    4:-0.5,    5:0,    6:0,    7:0,    8:1,  9:0 }


def get_reward_string():
    string_repr = "_"
    for key in dict_num_reward.keys():
        if dict_num_reward[key] != 0:
            string_repr +=  str(key) + str(dict_num_reward[key]).replace(".","") + "_"
    return string_repr[:-1]
    #--end for


model_weights_file_name = "weights_CNN_AE"
if RewardError: model_weights_file_name += "_RewErr"
if Resample: model_weights_file_name += "_Rsmpl"
if Sparsity: model_weights_file_name += "_Sprs"
if Array_Error: model_weights_file_name += "_ArrErr"
model_weights_file_name += "_" + "inOutType" + str(InputToOutputType)
model_weights_file_name += "_" + optimizer_type + "_" + str(batch_size)
model_weights_file_name += get_reward_string() + ".kmdl"

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

def keep_sample_by_reward():

    global cumulative_reward,x_train_target,x_train_original,x_train_reward,y_train
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
            prob_sampling = math.exp(-1*cumulative_reward*curr_reward)/math.exp(abs(cumulative_reward))
            # prob_sampling
            cutoff = np.random.rand()
            if cutoff < prob_sampling or RewardBasedResampling == False:
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

if train_model:
    print(" count positive reward samples = ", np.where(x_train_reward>0)[0].shape)
    print(" count negative reward samples = ", np.where(x_train_reward<0)[0].shape)
    print("average reward =", np.average(x_train_reward))

# #=============================
# if Resample and train_model:
#     new_x_train_indices = [keep_sample_by_reward(i, mnist_reward(y_train[i])) for i in range(x_train.shape[0])]
#     new_x_train_indices  = set(new_x_train_indices)
#     try:
#         new_x_train_indices.remove(-1)
#     except:
#         pass
# else: #dont resample
#     new_x_train_indices = range(x_train.shape[0])
# #=============================
# new_x_train_indices = list(new_x_train_indices)
# new_x_train_indices = new_x_train_indices[:int(len(new_x_train_indices)/1000)*1000] #for making sure we get batchsize divisible
# x_train_target = x_train[new_x_train_indices]
# x_train_original = x_train_original[new_x_train_indices]
# y_train = y_train[new_x_train_indices]
# x_train_reward = np.array([mnist_reward(y_train[i]) for i in range(len(y_train))])
# x_test_reward = np.array([mnist_reward(y_test[i]) for i in range(len(y_test))])

if not Negative_Error_From_Reward:
    x_train_reward = np.abs(x_train_reward)
    x_test_reward = np.abs(x_test_reward)

# encoding_dim = 32
input_img = Input(shape=(28,28,1))
target_img = Input(shape=(28,28,1))
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
    if Array_Error:
        xent_loss = error_function(target_img, decoded)
        reward_based_loss = input_reward_reshaped* xent_loss
    else:
        xent_loss = K.mean(error_function(target_img, decoded))
        reward_based_loss = input_reward * xent_loss

    autoencoder = Model(inputs=[target_img, input_img, input_reward], outputs=[decoded])
    autoencoder.add_loss(reward_based_loss)
    autoencoder.compile(optimizer=optimizer_type)
else:#not reward error
    if Array_Error:
        xent_loss = error_function(target_img, decoded)
    else:#not array error
        xent_loss = K.mean(error_function(target_img, decoded))

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
        autoencoder.fit([target_images,source_images,x_train_reward],epochs=num_epochs ,batch_size=batch_size,
                        shuffle=True,validation_data=([x_test,x_test,x_test_reward],None))
                # ,callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    else:
        autoencoder.fit([target_images,source_images],epochs=num_epochs ,batch_size=batch_size,
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


#find the indices of two of each class
# needed_numbers = [0,1,2,3,4,5,6,7,8,9]
needed_numbers = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
# needed_numbers = [i for i in dict_num_reward.keys() if dict_num_reward[i]>0]
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


#TODO CHANGE THE image to be overlayed on a static background, vs black background.
#very different from training data.
for i in target_indices:
    a_image = source_images[i]#KEEP it as 3-d (2-d + 1 b/w channel) for now
    image_mask = np.ma.make_mask(a_image)
    static_image = np.random.rand(x_train.shape[1], x_train.shape[2], x_train.shape[3])
    source_images[i] = source_images[i] + np.logical_not(image_mask)*static_image

import matplotlib.pyplot as plt
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
