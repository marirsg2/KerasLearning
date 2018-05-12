

import keras
import pandas
import matplotlib.pyplot as plt
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


dataframe = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
# plt.plot(dataframe)
# plt.show()
numpy.random.seed(7)
dataset = dataframe.values
dataset = dataset.astype('float32')
#normalize the dataset to between 0 to 1
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

'''
stateful: Boolean (default False). If True, the last state
for each sample at index i in a batch will be used as initial state
for the sample of index i in the following batch.
'''

#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

#todo relearn backprop in LSTMs

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
#+=====================================================+
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
#+=====================================================+

# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
#+=====================================================+
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(4,input_shape=(1,look_back))) #features, lookback.
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#yes we do allow shuffling, because here all we are using is ONE
#previous time step for prediction.LSTM is not really being used
#yet, this is just an example

#IMPORTANT: the input data is of the form [samplesxfeatures]. We could have more complex feature matrix like featuresXtime_steps
#we set batch size =1, BECAUSE we dont want the data in other batches to affect the current pass.
#we already have the timestep information in the input data in each sample.
# if you want to sample batches, then you could set batch=look_back, and shuffle = False
model.fit(trainX,trainY, epochs = 60, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()