from keras.datasets import mnist
import numpy as np
import random as rand
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import pickle
"""
:summary: take each image and add random numbers on either side of the number image. For example an "8" could be flanked
by compressed "3" and "7". or be flanked by 1 and 8 too.
"""





[x_train,y_train] , [x_test,y_test] = mnist.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
dimensions = (x_train.shape[1], x_train.shape[2])
# for i in range(len(x_train)): #This is for the original number with overlayed other numbers(UNSCALED)
smaller_data = []
for i in range(len(x_train)): #This is for the original number with overlayed other numbers(UNSCALED)
    if i%500 == 0:
        print(i)
    main_image = x_train[i]
    a_image = resize(x_train[np.random.randint(0,len(y_train))],(dimensions[0],dimensions[1]/2))
    b_image = resize(x_train[np.random.randint(0,len(y_train))],(dimensions[0],dimensions[1]/2))
    temp = np.concatenate((a_image,b_image),axis=1)
    mod_image = np.clip(main_image + temp, 0,1)
    x_train[i] = mod_image

for i in range(len(x_test)): #This is for the original number with overlayed other numbers(UNSCALED)
    if i%500 == 0:
        print(i)
    main_image = x_test[i]
    a_image = resize(x_test[np.random.randint(0,len(y_test))],(dimensions[0],dimensions[1]/2))
    b_image = resize(x_test[np.random.randint(0,len(y_test))],(dimensions[0],dimensions[1]/2))
    temp = np.concatenate((a_image,b_image),axis=1)
    x_test[i] = np.clip(main_image + temp, 0,1)



n=20 #number of images to be displayed
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_train[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(x_test[i])
    # a = source_images[target_indices[i]].reshape(source_images[0].shape[:-1])
    # b = decoded_imgs[target_indices[i]].reshape(source_images[0].shape[:-1])
    # final_mse = mse(a,b)
    # plt.title("mse=", str(final_mse))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

with open("compoundMnist.p","wb") as dest_file:
    pickle.dump(x_train,dest_file)
    pickle.dump(y_train,dest_file)
    pickle.dump(x_test,dest_file)
    pickle.dump(y_test,dest_file)