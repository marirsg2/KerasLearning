from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, RepeatVector
from keras.models import Model
from keras import regularizers
import numpy as np
import random as rand
import copy
from keras import backend as K
from keras import metrics
import math
import pickle
from sklearn.metrics import mean_squared_error as mse
from skimage.transform import rescale, resize, downscale_local_mean
import scipy.ndimage as image_reader
import os
import matplotlib.pyplot as plt

train_model = True
fraction_of_data = 1.0
image_dimensions =[176,176]
optimizer_type = 'sgd'
batch_size = 1
num_epochs = 20
error_function = metrics.mse
error_function_string = "mse"
min_num_data_points = 3000
RewardError = False
RewardBasedResampling = False
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
Predict_on_test = True
Resample = False
if RewardBasedResampling or Occlude or Invert_Img_Negative:
    Resample = True




model_weights_file_name = "CycleGAN_HorsesZebra_AE"
if RewardError: model_weights_file_name += "_RewErr"
if Resample: model_weights_file_name += "_Rsmpl"
if Sparsity: model_weights_file_name += "_Sprs"
if Array_Error: model_weights_file_name += "_ArrErr"
model_weights_file_name += "_" + "inOutType" + str(InputToOutputType)
model_weights_file_name += "_" + optimizer_type + "_" + str(batch_size)
model_weights_file_name += "_Dims" + str(image_dimensions)
model_weights_file_name += "_Error" + error_function_string
model_weights_file_name += ".kmdl"

# model_weights_file_name = "CNN_ae_weights_ResampleOcclude_NoiseToNoise_148.kmdl"
#=============================================
#prep the data
#get the Oranges data set and train an autoencoder.
train_dataset_source_B = "./CycleGAN_Dataset/datasets/horse2zebra/trainA/"
test_dataset_source_B = "./CycleGAN_Dataset/datasets/horse2zebra/testA/"
train_dataset_source_A = "./CycleGAN_Dataset/datasets/horse2zebra/trainB/"
test_dataset_source_A = "./CycleGAN_Dataset/datasets/horse2zebra/testB/"

train_source = train_dataset_source_A
test_source = test_dataset_source_B

x_train = []
for file_name in os.listdir(train_source):
    curr_nd_image = image_reader.imread(train_source + file_name, mode="L")
    curr_nd_image = resize(curr_nd_image,image_dimensions)
    curr_nd_image = np.reshape(curr_nd_image,image_dimensions+[1])
    x_train.append(curr_nd_image)
x_train = np.array(x_train)
x_train.astype('float32')/255
train_data_size = int(x_train.shape[0]/batch_size)*batch_size
x_train = x_train[0:train_data_size]


x_test = []
for file_name in os.listdir(test_source):
    curr_nd_image = image_reader.imread(test_source + file_name, mode="L")
    curr_nd_image = resize(curr_nd_image,image_dimensions)
    curr_nd_image = np.reshape(curr_nd_image, image_dimensions + [1])
    x_test.append(curr_nd_image)
x_test = np.array(x_test)
x_test.astype('float32')/255
test_data_size = int(x_test.shape[0]/batch_size)*batch_size
x_test = x_test[0:test_data_size]

print("train size=" , x_train.shape)
print("test size=" , x_test.shape)

# vizualize the image to confirm
# for single_im in x_test[-3:-1]:
#     single_im = np.reshape(single_im,image_dimensions)
#     plt.figure()
#     plt.imshow(single_im)
#     plt.gray()
#     plt.show()

x_train_original = copy.deepcopy(x_train)
x_train_target = x_train_original

# encoding_dim = 32
input_img = Input(shape=image_dimensions+[1])
target_img = Input(shape=image_dimensions+[1])
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
# x = MaxPooling2D((2,2),padding='same')(x)
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
x = Conv2D(16,(3,3),activation='relu',padding='same')(x)
# x = UpSampling2D((2,2))(x)
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



source_images = x_train_target
if Predict_on_test:
    source_images = x_test
target_indices = range(0,source_images.shape[0])


# score = autoencoder.evaluate([source_images,source_images])
# print(score.shape)
# print(score)

#TODO CHANGE THE image to be overlayed on a static background, vs black background.
#very different from training data.
# for i in target_indices: #THIS IS FOR SIDE BY SIDE TWO NUMBERS, mediocre to bad performance. Manages to filter 1 and 4 to some degree of success
#     a_image = resize(source_images[i].reshape(x_train.shape[1], x_train.shape[2]),(28,14))
#     b_image = resize(source_images[np.random.randint(0,len(y_target))].reshape(x_train.shape[1], x_train.shape[2]),(28,14))
#     # c_image = source_images[np.random.randint(0,len(y_target))].reshape(x_train.shape[1], x_train.shape[2])
#     # d_image = source_images[np.random.randint(0,len(y_target))].reshape(x_train.shape[1], x_train.shape[2])
#     temp = np.concatenate((a_image,b_image),axis=1).reshape(x_train.shape[1], x_train.shape[2],x_train.shape[3])
#     source_images[i] = temp

#
# for i in target_indices: #This is for the original number with overlayed other numbers(UNSCALED)
#     main_image = source_images[i]
#     a_image = resize(source_images[np.random.randint(0,len(y_target))].reshape(x_train.shape[1], x_train.shape[2]),(28,14))
#     b_image = resize(source_images[np.random.randint(0,len(y_target))].reshape(x_train.shape[1], x_train.shape[2]),(28,14))
#     # c_image = source_images[np.random.randint(0,len(y_target))].reshape(x_train.shape[1], x_train.shape[2])
#     # d_image = source_images[np.random.randint(0,len(y_target))].reshape(x_train.shape[1], x_train.shape[2])
#     temp = np.concatenate((a_image,b_image),axis=1).reshape(x_train.shape[1], x_train.shape[2],x_train.shape[3])
#     source_images[i] = np.clip(main_image + temp, 0,1)

# for i in target_indices: #This is to overlay multiple versions of the same number
#     main_image = source_images[i]
#     secondary_image = source_images[target_indices[np.random.randint(0,len(target_indices))]]
#     source_images[i] = np.clip(main_image + secondary_image, 0,1)

# for i in target_indices: #This is to overlay multiple numbers
#     main_image = source_images[i]
#     secondary_image = source_images[np.random.randint(0,len(source_images))]
#     source_images[i] = np.clip(main_image + secondary_image, 0,1)


# for i in target_indices: #This to add NOISE over the image. Decent performance in denoising and detecting feature.
#     main_image = source_images[i]
#     secondary_image = np.random.random((x_train.shape[1], x_train.shape[2],x_train.shape[3]))
#     secondary_mask =  np.random.randint(2,size=(x_train.shape[1], x_train.shape[2],x_train.shape[3]))#0 and 1
#     main_image = main_image * secondary_mask
#     source_images[i] = np.clip(main_image + secondary_image, 0,1)



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
