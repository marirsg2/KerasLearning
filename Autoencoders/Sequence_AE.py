from keras.layers import Input,LSTM,RepeatVector
from keras.models import Model

inputs = Input(shape=(timesteps,input_dim))#todo, if this is 2d, need flattening.
#input_dim is the size of the array. NOT 2d vector
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)#todo IMPORTANT. repeats for timestep number of times
decoded = LSTM(input_dim, return_sequences=True)(decoded)
#todo important #return sequences makes it many-to many

sequence_autoencoder = Model(inputs,decoded)
encoder = Model(inputs,encoded)