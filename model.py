import pandas as pd
import numpy as np
from scipy import misc
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
import math
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model

# Data augmentation constants
TRANS_X_RANGE = 10  # Number of translation pixels for augmented data (-RANGE/2, RANGE/2)
TRANS_Y_RANGE = 10  # Number of translation pixels for augmented data (-RANGE/2, RANGE/2)
TRANS_ANGLE = .3  # Maximum angle change when translating in the X direction
OFF_CENTER_IMG = .25  # Angle change when using off center images

DOWNSAMPLE_RATIO = 0.99
Learning_Rate = 0.0001
FOLDER = "examples/"
EPOCHS = 4
TRAINABLE = True
BRIGHTNESS_RANGE = 0.15
IMG_ROWS = 300
IMG_COLS = 300
SHAPE = (IMG_ROWS,IMG_COLS,3)

SAMPLES_TRAIN = 5000
SAMPLES_VALIDATION = 1000


def load_data(data):
	temp = []
	for i in range(len(data)):
		im = cv2.imread(data[i])
		im = misc.imresize(im,size=DOWNSAMPLE_RATIO)
		im = crop(im)
		# im = color_change(im)
		temp.append(im)
	return temp

def normalize(data):
	a=-0.5
	b=0.5
	greyscale_min=0
	greyscale_max=255
	return a + ( ( (data - greyscale_min)*(b - a) )/(greyscale_max - greyscale_min))

def color_change(data):
	x = cv2.cvtColor(data,cv2.COLOR_BGR2HSV)
	return x

def adjust_brightness(im):
	temp = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	# Compute a random brightness value and apply to the image
	brightness = BRIGHTNESS_RANGE * np.random.uniform(-1,1)
	temp[:, :, 2] = temp[:, :, 2] * (1-brightness)
	# Convert back to RGB and return
	return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)

def img_translate(img, angle):

	# Randomly form the X translation distance and compute the resulting steering angle change
	change = np.random.uniform(-0.5,0.5)
	x_translation = (TRANS_X_RANGE * change)
	new_angle = angle + (change * TRANS_ANGLE)

	# Randomly compute a Y translation
	y_translation = (TRANS_Y_RANGE * np.random.uniform(-0.5,0.5))

	# Form the translation matrix
	translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])

	# Translate the image
	return cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0])),new_angle


def crop(im):
	shape = np.array(im).shape
	y1 = int(shape[0]*0.4)
	y2 = int(shape[0]*0.87)
	# print(y)
	im = im[y1:y2 , : , :]
	im = cv2.resize(im, (IMG_ROWS, IMG_COLS), interpolation=cv2.INTER_AREA)
	return im

def curve_focus(xdata,ydata):
	count = 0
	for x in range(len(xdata)):
		if(ydata[x]==0.000):
			count+=1
	print("Total = {}\n0 Steering = {}".format(len(xdata),count))
	return xdata,ydata

def flip(xdata,ydata):
	for x in range(len(xdata)):
		flip = np.fliplr(xdata[x])
		xdata = np.append(xdata, [flip], axis=0)
		ydata = np.append(ydata, (-1*ydata[x]))
	return xdata,ydata

def set_model():
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
		input_shape=SHAPE,
		output_shape=SHAPE))
	model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
	model.add(Convolution2D(36,5,5,border_mode='same',activation="elu",
	 name='conv1'))
	model.add(Convolution2D(48,3,3,activation="elu",border_mode='same',
	 name='conv2'))
	model.add(Convolution2D(64,3,3,activation="elu",border_mode='same', 
		name='conv3'))
	model.add(Convolution2D(64,3,3,activation="elu",border_mode='same', name='conv4'))
	model.add(Flatten(name='flat1'))
	# model.add(Dropout(0.2))
	# model.add(Dense(1164, activation="elu"))
	# model.add(Dropout(.3, name='drop1'))
	model.add(Dense(100, activation="elu", name='dense1'))
	model.add(Dense(50, activation="elu", name='dense2'))
	model.add(Dense(10, activation="elu", name='dense3'))
	model.add(Dense(1, activation="linear", name='dense4'))
	return model

def my_range(start, end, step):
	while start <= end:
		yield round(start,1)
		start += step

def show_data(log):
	fig = plt.figure(figsize=(8,2))
	a = fig.add_subplot(1,2,1)
	im = cv2.imread(FOLDER+log[560,0].strip())
	im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
	a.set_title("Full Resolution")
	plt.axis('off')
	plt.imshow(im)
	im = misc.imresize(im,size=0.2)
	a = fig.add_subplot(1,2,2)
	a.set_title("After 80% Downsampling")
	plt.imshow(im)
	# im = crop(im)
	# im, an = process_line(log[600])
	# a = fig.add_subplot(2,1,2)
	# im, an = process_line(log[600])
	# plt.imshow(im,aspect="auto",interpolation="nearest")
	plt.axis('off')
	fig.savefig('examples/Downsampling.png')
	plt.show()
	exit()
	# plt.hist(steer,bins=100)
	# plt.show()
	# exit()
	count = 1
	y = 0
	steer = log[:,3]
	for x in my_range(-0.8,0.7,0.1):
		while 1:
			y = np.random.randint(len(steer))
			if(round(steer[y],1)==x):
				print("Found {}",(x))
				break
			# else:
			# 	print("Discarded {}",steer[y])
		a=fig.add_subplot(4,5,count)
		im = cv2.imread(FOLDER+log[y,0])
		im,angle = process_line(log[y])
		a.set_title(str(x)+" to "+str(round(angle,1)))
		plt.imshow(im,aspect="auto",interpolation="nearest")
		count+=1
		# print(x)
	plt.show()

	exit()
	pic = np.random.randint(len(X_train))
	print(X_train.shape)
	plt.imshow(X_train[pic])
	plt.show()
	exit()

def augment(x,y):
	x,y = flip(x,y)
	return x,y

def process_line(sample):

	img_choice = np.random.randint(3)	

	angle = 0.0
	if(img_choice==0):
		angle = float(sample[3])
	elif(img_choice==1):
		angle = float(sample[3])+0.27
	elif(img_choice==2):
		angle = float(sample[3])-0.27

	im = cv2.imread(FOLDER+sample[img_choice].strip())
	im = misc.imresize(im,size=DOWNSAMPLE_RATIO)
	im = crop(im)
	im = adjust_brightness(im)
	im,angle = img_translate(im,angle)
	# im = normalize(im)

	return im,angle

def generator(samples, batch_size=32):
	"""
	Purpose: Yield tensor batches to fit_generator function
	Inputs: A file path
	Outputs: X_train, a [AHH, 80, 320, 3] tensor and y_train, a [AHH, 1] matrix
	Where AHH = ((FEATURE_GENERATION_MULTIPLE * 3) + 3) * BATCH_SIZE
	"""
	num_samples = len(samples)
	shuffle(samples)
	while 1: # Loop forever so the generator never terminates
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				image,angle = process_line(batch_sample)
				images.append(image)
				angles.append(angle)

			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			X_train, y_train = augment(X_train,y_train)
			yield shuffle(X_train, y_train)


if __name__ == "__main__":
	log = pd.read_csv(FOLDER+"driving_log.csv").values
	show_data(log)
	

	print(log.shape)
	train_samples, validation_samples = train_test_split(log,test_size=0.2)
	
	im,an = process_line(train_samples[np.random.randint(len(train_samples))])
	print(np.array(im).shape)
	# plt.imshow(im)
	# plt.title(str(an))
	# plt.show()
	# exit()
	model = set_model()
	# model.load_weights('weights.h5',by_name=True) 

	adam = Adam(lr=Learning_Rate)
	model.compile(optimizer = adam, loss = 'mean_squared_error')
	history=model.fit_generator(generator(train_samples), samples_per_epoch = 
			SAMPLES_TRAIN, validation_data=generator(validation_samples), 
			nb_val_samples=SAMPLES_VALIDATION, nb_epoch=EPOCHS, verbose=1)
	model.save_weights('weights.h5')
	model.save('model.h5')
	
	print("Model saved")

def for_drive(im):
	print(im.shape)
	x = im
	x = misc.imresize(x,size=DOWNSAMPLE_RATIO)
	x = crop(x)
	# plt.imshow(x)
	# plt.show()
	# x = color_change(x)
	# x = normalize(x)
	return x

