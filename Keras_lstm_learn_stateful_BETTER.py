
# LSTM for international airline passengers problem with memory
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


#to save and restore lstm states. useful when you want to predict and fit in tandem, especially if you need more than 1 step.
import keras.backend as K
def get_model_states(model):
    return [K.get_value(s) for s,_ in model.state_updates]
def set_model_states(model, states):
    for (d,_), s in zip(model.state_updates, states):
        K.set_value(d, s)


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0] #0 is the feature dimension.
		# here there is only one feature. If there were more, we would need all of them
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0]) # the target is the NEXT one for this problem. Predict next in sequence
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 1 #yes lookback of 4 is weaker than 1, only in this case. BUT you need larger lookback to do BPTT
#PAY ATTENTION to how the data set is created to allow this. And how the states are being preserved.
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# create and fit the LSTM network
batch_size = 1
#TODO READ how the data is saved BEFORE the lstm model. SEE BATCH SIZE = 1
#TODO read the code for how the data set is created for lookback in "create dataset"
model = Sequential()
model.add(TimeDistributed(Dense(5), batch_input_shape=(batch_size, look_back, 1)))
# model.add(Dense(5, batch_input_shape=(batch_size, look_back, 1)))
model.add(LSTM(4, stateful=True)) #batch_input_shape=(batch_size, look_back, 1)
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print (model.summary())

for i in range(100):

	#batch size is 1 for stateful
	# is repeating many of the previous steps and is thus changing the state.
	model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
	#todo note that the state and the edge weights are tuned to each other. resetting one and not the other can result in nonsense.
	#IT SEEMS as though you can safely predict , restore state and train.
	prev_state = get_model_states(model)
	testPredict = model.predict(testX, batch_size=batch_size) #todo note that predict does NOT affect the weights, duh, only the state, so can recover
	set_model_states(model, prev_state)#recover state
	testPredict = model.predict(testX, batch_size=batch_size)
	#---end prediction step
	model.reset_states()
# make predictions
trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()
testPredict = model.predict(testX, batch_size=batch_size)
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