import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.model_selection import train_test_split
import time
import cv2

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model

DOWNSAMPLE_RATIO = 0.2
Learning_Rate = 0.0001
FOLDER = "examples/"
EPOCHS = 1

def remove_first_char(x):
	return x[1:]

def load_data(data):
	temp = []
	for i in range(len(data)):
		im = misc.imread(data[i])
		im = misc.imresize(im,size=DOWNSAMPLE_RATIO)
		im = crop(im)
		im = gray(im)
		temp.append(im)
	return temp

def normalize(data):
    a=-0.5
    b=0.5
    greyscale_min=0
    greyscale_max=255
    return a + ( ( (data - greyscale_min)*(b - a) )/(greyscale_max - greyscale_min))

def gray(data):
	x = cv2.cvtColor(data,cv2.COLOR_BGR2YUV)
	x = x[:,:,0:1]
	return x

def crop(data):
	shape = data.shape
	y1 = int(shape[0]*0.3)
	y2 = int(shape[0]*0.87)
	# print(y)
	return data[y1:y2 , : , :]

def curve_focus(xdata,ydata):
	count = 0
	for x in range(len(xdata)):
		if(ydata[x]==0.000):
			count+=1
	print("Total = {}\n0 Steering = {}".format(len(xdata),count))
	return xdata,ydata

def flip(xdata,ydata):
	for x in range(len(xdata)):
		xdata.append(np.fliplr(xdata[x]))
		ydata = np.append(ydata,(-1*ydata[x]))
	return xdata,ydata

def set_model():
	model = Sequential()
	model.add(Convolution2D(36,5,5,border_mode='valid',activation="elu",
		input_shape=X_train[0].shape), trainable="False")
	model.add(Convolution2D(48,3,3,activation="elu",border_mode='valid'),trainable="False")
	model.add(Convolution2D(64,3,3,activation="elu",border_mode='valid'),trainable="False")
	model.add(Convolution2D(64,3,3,activation="elu",border_mode='valid'))
	model.add(Flatten())
	# model.add(Dropout(0.5))
	# model.add(Dense(300, activation="elu"))
	model.add(Dense(100, activation="elu"))
	model.add(Dense(50, activation="elu"))
	model.add(Dense(10, activation="elu"))
	model.add(Dense(1, activation="linear"))
	return model

def for_drive(im):
	print(im.shape)
	x = im
	x = misc.imresize(x,size=DOWNSAMPLE_RATIO)
	x = crop(x)
	x = gray(x)
	x = normalize(x)
	return x
# columns = ['center', 'left', 'right', 'steering','throttle','brake','speed']
if __name__ == "__main__":
	log = pd.read_csv(FOLDER+"driving_log.csv")
	# print(type(load_data((for s in log.ix[:,1:2].values[:100]: return s)))
	start = time.time()

	# Load data
	X_train = load_data(log.iloc[:,0])
	X_train += load_data(log.iloc[:,1].str.lstrip())
	X_train += load_data(log.iloc[:,2].str.lstrip())
	y_train = log.ix[:,3:4].values
	y_train = np.append(y_train,log.ix[:,3:4].values+0.27)
	y_train = np.append(y_train,log.ix[:,3:4].values-0.27)

	print("Time taken = ",time.time()-start)
	# plt.imshow(X_train[0][:,:,0],cmap="gray")
	# plt.show()
	# Process data
	# X_train,y_train = curve_focus(X_train,y_train)
	X_train,y_train = flip(X_train,y_train)
	X_train = np.array(X_train)
	# plt.hist(y_train)
	# plt.show()
	X_train = normalize(X_train)
	
	# Splitting data
	print(X_train.shape)
	X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,
		test_size=0.15,random_state=0)
	print(X_train.shape)

	# model = set_model()
	model = load_model('model1.h5') 

	adam = Adam(lr=Learning_Rate)
	model.compile(optimizer = adam, loss = 'mean_squared_error', 
		metrics = ['mean_squared_error'])
	history = model.fit(X_train, y_train, batch_size=30, nb_epoch=EPOCHS, 
		validation_data=(X_validation,y_validation))
	model.save('model1.h5')
	print("Model saved")
