#**Behavioral Cloning** 


---

**Behavioral Cloning Project**

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
* README.md summarizing the results. 

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed.

My model takes inspiration from the [NVIDIA model](https://arxiv.org/pdf/1604.07316v1.pdf)  mentioned in the lectures.  
The model consists of five CNNS and three fully connected layers. 

The model includes RELU layers in between each CNN to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer (code line 85). 

####2. Attempts to reduce overfitting in the model

The model contains  L2 regularizationin all CNNs and fully connected networks. 

Adam optimizer is used with the loss function being mean squared error (MSE). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 116).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the [NVIDIA model](https://arxiv.org/pdf/1604.07316v1.pdf). I thought this model might be appropriate because the paper had positive results with an autonomy index of 98%

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to add regularisation to each of CNN and fully connected layers

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The model consists of three convolution neural (31x98, 14x47, 5x22) with padding of 2x2 and depths of 24, 36 and 48 respectively (model.py lines 88-93). This is followed by two CNNS (3x20, 1x18) with a depth of 64 with no strides. The CNNs are followed by three Fully connceted layers 


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one and half laps using center lane driving. I also used the data provided by udacity. Here is an example image of center lane driving:
Also I load the left, right and center camera images while adjusting the steering angle (+0.25 for the left frame and -0.25 for the right).
![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles with a probability of 0.5. This would help the vehicle to recover from the other direction since the track is driven counterclockwise. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 28347) number of data points. I then preprocessed this data by resizing the images as required by the model. Then I added brightness of images and introducing shadows in parts of images.


For the validation data I used the raw images and avoided preproccing the images. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by decreasing training and validation losses. I used an adam optimizer so that manually training the learning rate wasn't necessary.
