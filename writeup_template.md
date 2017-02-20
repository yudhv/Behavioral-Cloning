#**Behavioral Cloning** 

##Brief Description

###In this project, we are given a simulated environment of a test track, where we drive a car manually at first and let a Convolutional Neural Network model learn from our driving "behavior". The model is only given images from the simulator as training data. Finally, the model is given a chance to drive the track itself without any human intervention. Let's begin!!!

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 
```sh
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=SHAPE, output_shape=SHAPE))
model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
model.add(Convolution2D(36,5,5,border_mode='same',activation="elu",name='conv1'))
model.add(Convolution2D(48,3,3,activation="elu",border_mode='same',name='conv2'))
model.add(Convolution2D(64,3,3,activation="elu",border_mode='same', name='conv3'))
model.add(Convolution2D(64,3,3,activation="elu",border_mode='same', name='conv4'))
model.add(Flatten(name='flat1'))
model.add(Dense(100, activation="elu", name='dense1'))
model.add(Dense(50, activation="elu", name='dense2'))
model.add(Dense(10, activation="elu", name='dense3'))
model.add(Dense(1, activation="linear", name='dense4'))
```

Note: the model includes RELU activations in all layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

In-order to reduce overfitting, the following steps were employed:
* Using lower resolution images - the images used were downsampled to almost 10% of their original sizes. This only kept the basic features of the road, lane lines, and side fencing to remain visible. The unnecessary part, i.e the scenery, grounds, and water bodies were all reduced to mere colors. This helped greatly in removing any chances of overfitting, while keeping the test data manageable and trainable even on a discreet GPU-less machine such as the one I had. (Insert image)
* Using a lower learning rate - in this project, I used a low learning rate 1e-4 as it wasn't small enough to cause underfitting in the limited epochs I trained the model for, nor was it too high to cause local minima situations. (Insert bridge images here)
* Using a smaller set of images - only 8,000 images were used to train the model. The images were taken from Udacity's test data, which had smoother steering angles than the ones I could generate from a keyboard during manual driving (Insert the heart chart here)

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
