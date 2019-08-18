# **Behavioral Cloning** 

## Writeup 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center Image"
[image2]: ./examples/left.jpg "Left Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used LeNet for the first try and then applied Nividia's CNN model which is better.

#### 2. Attempts to reduce overfitting in the model
 
 The model was trained and validated on different data sets to ensure that the model was not overfitting. 
 The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road .

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The first step before building the model is to collect data. The details about how I collect is in section3.
After finishing data collection, I used the CNN architecture created by Nvidia (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) because this model is better than LeNet after doing some experiment.
I split the data into training set and validation set to avoid overfitting. After training the model, it has a low mean squared error on both the training and validation sets.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track which occurs only on the last two turns. In order to improve the driving behavior in these cases, I collect more turning data only for these two turns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

This is the architecture of the model:
* A Lambda layer to parallelize image normalization. It normalize all the pixel values within the range [-1, 1]
* A Cropping layer which crops the image size to 70x25
* A Convolutional layer with 5x5 kernel, stride of 2, depth of 24 followed by RELU activation
* A Convolutional layer with 5x5 kernel, stride of 2, depth of 36 followed by RELU activation
* A Convolutional layer with 5x5 kernel, stride of 2, depth of 48 followed by RELU activation
* A Convolutional layer with 3x3 kernel, depth of 64 followed by RELU activation
* A Convolutional layer with 3x3 kernel, depth of 64 followed by RELU activation
* A Fully-connected layer with 100 hidden units
* A Fully-connected layer with 50 hidden units
* A Fully-connected layer with 10 hidden units
* A Fully-connected layer with 1 hidden units (steering angle output)


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

Next step is to simulate the recovery laps, I only record data when the car is driving from the side of the road back toward the center line. This is an example when the car is driving in the left side

![alt text][image2]

To augment the data sat, I also flipped images and angles thinking that this would help generalize the model. 

One thing to keep in mind is that cv2.imread reads the image in BGR color format, I need to convert it into RGB color format becauses drive.py script reads in the data in RGB format.

After the collection process, I had 40388 number of data points, the dimension of each image is 320x160x3

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 after experiment. I used an adam optimizer so that manually training the learning rate wasn't necessary.
