

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
# print keras.__version__

np.random.seed(123)#for reproducibility

(X_train,y_train),(X_test,y_test) = mnist.load_data()
# print X_train.shape
# plt.imshow(X_train[0])
# plt.show()

#========PREPROCESS RAW DATA FOR KERAS=================
#the depth (number of channels for ConvNets) is 1 for the MNIST data
# print X_train.shape
X_train = X_train.reshape((X_train.shape[0],28,28,1))
X_test = X_test.reshape((X_test.shape[0],28,28,1))
# print X_train.shape


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#========PREPROCESS LABELS DATA FOR KERAS=================
# print(y_train.shape)
Y_train = np_utils.to_categorical(y_train,10)
Y_test = np_utils.to_categorical(y_test,10)
# print(y_train.shape)
#====================KERAS MODEL defs============================

model = Sequential()

model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
print("The first conv hidden layer shape is = ", model.output_shape)
model.add(Convolution2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
print("The hidden layer shape after first max pool and dropout is = ",\
      model.output_shape)

model.add(Flatten())
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation = 'softmax'))
#=================COMPILE THE MODEL====================

model.compile(loss = 'categorical_crossentropy',
               optimizer = 'adam',
              metrics = ['accuracy'])

#====================TRAIN MODEL============================

model.fit(X_train,Y_train,batch_size=32,nb_epoch=10,verbose=1)

#====================TEST MODEL============================
score = model.evaluate(X_test,Y_test,verbose = 0)

#====================TEST MODEL============================

print ("end")


