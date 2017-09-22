**Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/dist_final.png "Data distribution"
[image2]: ./images/center_lane_driving.jpg "Center lane driving"
[image3]: ./images/recovery_from_right.jpg "Recovery Image (from right)"
[image5]: ./images/recovery_from_left.jpg "Recovery Image (from left)"
[image6]: ./images/before_flipped.jpg "Normal Image"
[image7]: ./images/flipped.jpg "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results. 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed.

My model takes inspiration from the [NVIDIA model](https://arxiv.org/pdf/1604.07316v1.pdf)  mentioned in the lectures.  
The model consists of five CNNS and three fully connected layers. 

The model includes RELU layers in between each CNN to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer (code line 77). 

#### 2. Attempts to reduce overfitting in the model

The model contains  L2 regularizationin all CNNs and fully connected networks. 

Adam optimizer is used with the loss function being mean squared error (MSE). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 108).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with the simplest model and try building on it. I started  with just 1 fully connected network to set up the model. Then I tried using the Lenet from the previous assignment. The model did not give satisfactory results and therefore I tried the  [NVIDIA model](https://arxiv.org/pdf/1604.07316v1.pdf) as suggested in the lectures. I thought this model might be appropriate because the paper had positive results with an autonomy index of 98%. I ignored the 1164 depth dense layer in the NVIDIA model as it caused memory problems on AWS g2.2x server. 
 

The final model looks like this


Layer (type)                     Output Shape                           

lambda_1 (Lambda)                (None, 66, 200, 3)          
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 33, 100, 24)
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 33, 100, 24)
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 50, 36)
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 17, 50, 36) 
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 9, 25, 48)   
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 9, 25, 48)  
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 9, 25, 64)
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 9, 25, 64) 
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 9, 25, 64)
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 9, 25, 64)  
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 14400)      
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)         
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 100)         
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)         
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 50)           
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 10)            
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             



In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to add regularisation to each of CNN and fully connected layers


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. One such place was when the vehicle had to take a clockwise turn. Since the model was trained on a data where the vehicle predominently made anti clockwise turns, this had caused a problem. TO overcome this random images were flipped (along with angles) before training.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The model consists of three convolution neural with 5x5 kernel size(o/p size 31x98, 14x47, 5x22) with padding of 2x2 and depths of 24, 36 and 48 respectively (model.py lines 88-93). This is followed by two CNNS with 3x3 kernel(o/p size 3x20, 1x18) with a depth of 64 with no strides. The CNNs are followed by three Fully connceted layers 


#### 3. Creation of the Training Set & Training Process

The data was skewed over zero driving angle and therefore certain images were deleted depending on the ratio of data distribution. The final data distribution after processing looks like this
![alt text][image1]


To capture good driving behavior, I first recorded one lap on track two laps using center lane driving.I also recorded a lap where the car had to take a lot of recovery turns to get bck to the center of lane I also used the data provided by udacity. Also I load the left, right and center camera images while adjusting the steering angle (+0.25 for the left frame and -0.25 for the right). Here is an example image of center lane driving:

![alt text][image2]



I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return to center if it goes off track. These images show what a recovery looks like.

![alt text][image3]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles with a probability of 0.5. This would help the vehicle to recover from the other direction since the track is driven counterclockwise. For example, here are the images before and after being flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 28347) number of data points. I then preprocessed this data by resizing the images as required by the model. Then I added brightness of images and introducing shadows in parts of images.


For the validation data I used the raw images and avoided preproccing the images. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by decreasing training and validation losses. I used an adam optimizer so that manually training the learning rate wasn't necessary.
