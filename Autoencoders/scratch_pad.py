# K.tensorflow_backend._get_available_gpus()
from keras import metrics
from keras import backend as K
import numpy as np

def squared_error(a,b):
    return K.square(a-b)

a = K.random_normal_variable(shape=(3,4), mean=0, scale=1)
b = K.random_normal_variable(shape=(3,4), mean=0, scale=1)

c = squared_error(a,b)

print(c)


