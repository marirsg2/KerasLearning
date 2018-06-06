# K.tensorflow_backend._get_available_gpus()
from keras import metrics
from keras import backend as K


a = K.random_normal_variable(shape=(3,), mean=0, scale=1)
b = K.random_normal_variable(shape=(3,), mean=0, scale=1)

c = metrics.mse(a,b)

print(c)


