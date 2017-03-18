# **Behavioral Cloning** 

## Dependencies

This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The following resources can be found in this github repository:
* drive.py
* video.py
* model.py

The following command must be used in the project's root directory to see the Neural Network in action!!
```sh
python drive.py model.h5
```
Udacity's simulator must be installed and running in Automous mode. 

Simulator download links-
[Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)
[Mac](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)
[Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)

## Brief Description

#### In this project, we are given a simulated environment of a test track, where we drive a car manually at first and let a Convolutional Neural Network model learn from our driving "behavior". The model is only given images from the simulator as training data. Finally, the model is given a chance to drive the track itself without any human intervention. Let's begin!!!

The goals / steps of this project are the following:
* Using a simulator to collect data of good driving behavior
* Building a convolution neural network in Keras that predicts steering angles from images
* Training the model with randomly split training and validation sets
* Testing the model by letting it successfully drive around a simulated track autonomously

[//]: # (Image References)

[image1]: ./examples/heart_rate_chart_bad.png "Smoothness image custom data"
[image2]: ./examples/Downsampling.png "Downsampling"
[image3]: ./examples/all_new.png "Aggregation after"
[image4]: ./examples/all_old.png "Aggregation before"
[image5]: ./examples/heart_rate_chart_good.png "Smoothness image udacity data"
[image6]: ./examples/hist_new.png "Histogram after"
[image7]: ./examples/hist_old.png "Histogram before"
[image8]: ./examples/model.png "Model"
[image9]: ./examples/recovery1.jpg "recovery1"
[image10]: ./examples/recovery2.jpg "recovery2"
[image11]: ./examples/recovery3.jpg "recovery3"
[image12]: ./examples/flip.png "Flip"

---

## Detailed Analysis
### DATA
#### Which data to choose?

The training data derived by manually driving through Track 1 was very abrupt. This is especially true if you use a keyboard to drive rather than a mouse or a joystick, which was the case for me as the MacOS simulator does not allow mouse input for steering angles. Here is a comparison of the official Udacity provided data and the one I recorded using a keyboard on Track 1 - 

*Smooth angle distribution*
![image5]

*Non-smooth angle distribution*
![image1]

As can be seen quiet clearly from the steering angle spikes, the Udacity data is much more smoother than the one obtained through keyboard based inputs. This adversely affects the training phase where the model only learns how to drive "rash", and hence is predisposed to naturally going out of the road because of the random turn in a wrong direction. This is why I had no option but to use the limited Udacity data for training. 

However, all is not finished yet. To make the most of this data, I used a combination of artificial techniques to augment and balance the data, like randomly choosing between center, left, and right camera images (and adjusting steering angles accordingly), X and Y axis translation, image flipping, normalization, cropping, brightness adjustment, etc. 

For details about how I created the training data, skip to the next section. 

#### Reducing Overfitting

In-order to reduce overfitting, the following steps were employed:
* Using lower resolution images - the images used were downsampled to almost 10% of their original sizes. This only kept the basic features of the road, lane lines, and side fencing to remain visible. The unnecessary part, i.e the scenery, grounds, and water bodies were all reduced to mere colors. This helped greatly in removing any chances of overfitting, while keeping the test data manageable and trainable even on a discreet GPU-less machine such as the one I had. 
* Using a lower learning rate - in this project, I used a low learning rate 1e-4 as it wasn't small enough to cause underfitting in the limited epochs I trained the model for, nor was it too high to cause a situation with local minima.
* Using a smaller set of images - only 8,000 images were used to train the model. The images were taken from Udacity's test data, which had smoother steering angles than the ones I could generate from a keyboard during manual driving.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### MODEL ARCHITECTURE

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

Note: the model includes RELU activations in all layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### Solution Design Approach

The overall strategy for deriving a model architecture was to go from an underfitting model to one that was just starting to overfit. In the interim, I believed I would find the perfect solution. 

My first step was to use a convolution neural network model similar to the one mentioned in Nvidia's brilliant [End to End Learning whitepaper](Screen Shot 2017-02-21 at 3.06.43 PM). I thought this model to be appropriate because Nvidia aimed to solve a problem that was remarkably similar to the one I was facing. They tried to guide a car simply image-based data without giving their autonomous car any pre-trained logic to rely on, and trusted their Deep Learning network to figure out the important aspects of image data just like a teenager would while trying to learn how to drive for the first time. The idea resonated with me and I set about to base my strategy on Nvidia's solid foundation. 

Consequently, I built a model very similar to the one Nvidia used. It had a starting normalization layer followed by Convolutional layers with RELU activation. Finally, 5 Fully connected (FC) layers were used that converged gradually to give a single steering angle output in the end.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To mitigate the overfitting, I made the following changes-
* Removing the first fully connected layer with 1164 outputs and instead, going straight with the 100 unit wide 2nd FC layer. This heavily reduced my RAM consumption by a factor of 8.
* Removing the 3rd Convolutional Layer altogether
* Performing data augmentation and processing techniques to make the model more robust. These included flipping images, X and Y translation, brightness adjustment, downscaling, and cropping. 

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### Model parameter tuning

The model used an adam optimizer with a learning rate of 1e-04. The number of epochs was set at 5, as there wasn't a substantional observed improvement after. 

#### Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes -
![image8]

* The first layer is tasked with normalizing the images to values between -1 and 1
* The Convolution layer is based out of a [brilliant idea](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.lvotuk68j) where the model learns to select which of the three color spaces is best for achieving minimum loss
* 3, 4, 5, and 6 Convolution layers are meant to perform the usual feature extractions from the image data in increasing complexity. The final convolutional layer has a depth of 64 filters and output size 30 * 30
* The 4 Fully Connected layers help converge the model to a single output using linear activation in the last FC layer. This output is then used as the steering angle prediction of the model. 

### TRAINING

To capture good driving behavior, I used Udacity's "smoother" driving data that had a combination of center lane driving and recovery recordings. 

A note on Recovery recording - If our model only learns from center lane driving, then there is a high likelihood that it would not know what to do once the car moves away from the center of the road (which is likely to happen since we never achieve a 0 loss during pratical training). In such a case, it is imperative to teach the model how to recover from probable failures (e.g when the car is moving towards the side lanes). Hence, recordings have been done in such a way where the car is observed to be at the very edge of the road in the start, and then drives back to the middle of the road. Here is an example - 

![image9]

![image10]

![image11]

Although Udacity's data was smoother than what one could generate through a keyboard, the data still wasn't balanced. It had a much higher percentage of 0 steering angle than any other. This was bound to make the model learn driving "straight" and ended up causing problems during sharp turns, where the car would simply run out of the road due to its straight line driving. 

![image7]

Coming back to the training data, the following steps were performed to augment and process the data - 
* Flipping images - To augment the data sat, I also flipped images and angles thinking that this would not only add some much needed generalization to the model, but would also not add any noise to the training data. This was a simple and effective technique to double the training set. Here is an example -
![image12]
* X and Y translation - Moving the image a random number of pixels in the X direction was equivalent to making the model feel that the car was moving on the road. Coupled with this was an equivalent amount of change in the steering angle that helped generalize the model better
* Brightness - Changing the brightness aspect of the images led the model to be more resistant to lighting effects like shadows, time of day, and weather. This, I believe, will lead to better performance in other environments as well. 
* Downscaling and squaring images - Downscaling the images was one of the biggest factors that helped the model generalize even with the limited data samples I had at my disposal. My assumption is that with low res images, the model was unable to recognize "objects" from the surrounding environments and hence, did not get the chance to overfit. However, I'll have to confirm this hypothesis by looking at the outputs of individual layers. Also, resizing the images to a square resolution of 30 * 30 * 3 helped the 5 * 5 and 3 * 3 filters better fit into the data without needing to add unnecessary padding. 
![image2]

After the augmentation and balancing process, I had ~16000 data points which looked like this

![image6]

At first comparison, it becomes evident that the data is much better now. It is less biased toward a 0 steering angle, has double the number of data points, and is much more robust to changes in lighting and the relative position of the car on the road.

Here is another comparison - 

*Old data*

![image4]

*Processed data*

![image3]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

---

### Conclusion
On testing the above model on the simulator, the car drove indefinitely without touching the sidelanes. The goal was achieved and it was proven that a dataset as small as 8,000 images can be enough to successfully train a Convolutional Neural Network. This was also the smallest dataset used among my fellow Udacity students to train and test a CNN successfully on the simulation. 

[Here is a link to the final output](examples.mp4)

#### What's next?
There is a Track 2....
In addition to covering that, I'd like to add lane detection and object detection to this code as well. I've already implemented those as separate projects [here](https://github.com/yudhvir/Advanced-Lane-Finding) and [here](https://github.com/yudhvir/Vehicle-Detection), but when combined, these three projects could help make a Level 3 self driving car. 
Now that would be pretty awesome!!!
