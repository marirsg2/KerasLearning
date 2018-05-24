import numpy as np
from sklearn.metrics import mean_squared_error as mse
#
# a = np.array([[1.3,2.4,3.7],[2.1,3.4,2.1]])
# b = np.array([[1.3,2.4,3.7],[2.1,3.4,2.2]])
#
# c = mse(a,b)
# print(str(c))


# import tensorflow as tf
#
#
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

from keras import backend as K
K.tensorflow_backend._get_available_gpus()