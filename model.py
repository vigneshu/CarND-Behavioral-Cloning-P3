from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.utils import shuffle
from keras.layers.core import Dense, Activation, Flatten, Lambda
import math
import numpy as np
from PIL import Image         
from sklearn.model_selection import train_test_split
import cv2                 
from os import getcwd
import csv
import tensorflow as tf
import random
import scipy.ndimage
import matplotlib.pyplot as plt
def preprocess_image(img, validation=False):
    '''
    resize images as required by th pipeline
    add brightness to images
    introduce shadows to parts of image
    '''
    processed_img = img[50:140,:,:]
    processed_img = cv2.GaussianBlur(processed_img, (3,3), 0)
    processed_img = cv2.resize(processed_img,(200, 66), interpolation = cv2.INTER_AREA)
    

    if validation:
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2YUV)
        return processed_img

    # brightness
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
    processed_img[:,:,0] = processed_img[:,:,0]*np.random.uniform(0.8, 1.2) 
    processed_img = cv2.cvtColor(processed_img,cv2.COLOR_HSV2BGR)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2YUV)  

    processed_img = img.astype(float)
    # add shadow in images
    h,w = processed_img.shape[0:2]
    start_rand = np.random.randint(0,w)
    end_rand = np.random.randint(start_rand,w)
    processed_img[:,start_rand:end_rand,0] *= np.random.uniform(0.6,0.8)
    processed_img = cv2.resize(processed_img,(200, 66), interpolation = cv2.INTER_AREA)
    return processed_img


def get_data(image_path, angles, batch_size=64, validation=False):
    '''
    preprocess the images
    flip 50 % of images
    '''
    X, y = ([],[])
    image_path, angles = shuffle(image_path, angles)
    while True:       
        for i in range(len(angles)):
            img = cv2.imread(image_path[i])
            angle = angles[i]
            img = preprocess_image(img, validation)
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                angle *= -1

            X.append(img)
            y.append(angle)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([],[])
                image_path, angles = shuffle(image_path, angles) 


def get_model():
    model = Sequential()

    # Normalize
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))
    # Add three CNN with kernel size 5,5 and two with kernel size 3,3
    # Output of the CNN will have dimensions  31x98, 14x47, 5x22 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Convolution2D(24, 5,5, subsample=(2, 2), border_mode='same', W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 15,5, subsample=(2, 2), border_mode='same', W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5,5, subsample=(2, 2), border_mode='same', W_regularizer=l2(0.001)))
    model.add(Activation('relu'))

    # Add  3x20, 1x18 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(0.001)))
    model.add(Activation('relu'))

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers (depth 1164,100, 50, 10), tanh activation (and dropouts)
    # model.add(Dense(1164))
    # model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))

    # Add a fully connected output layer
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    return model

'''
Main program 
'''
# home_dir = [getcwd() + '/1lap/IMG/', getcwd() + '/custom_data/IMG/', getcwd() + '/data/IMG/']
# csv_path = ['./1lap/driving_log.csv','./custom_data/driving_log.csv','./data/driving_log.csv']
home_dir = [ getcwd() + '/custom_data/IMG/', getcwd() + '/data/IMG/']
csv_path = ['./custom_data/driving_log.csv','./data/driving_log.csv']
image_path = []
angles = []

'''
read the CSV file, modify path of images since eremote aws is used to train network
'''
for i,path in enumerate(csv_path):
    with open(path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            try:
                if float(line[6]) < 0.1 :
                    continue
                image_path.append(home_dir[i] + line[0].split("/")[-1])
                angles.append(float(line[3]))
                image_path.append(home_dir[i] + line[1].split("/")[-1])
                angles.append(float(line[3])+0.25)
                image_path.append(home_dir[i] + line[2].split("/")[-1])
                angles.append(float(line[3])-0.25)
            except:
                pass




image_path = np.array(image_path)
angles = np.array(angles)
bin_count = 23
avg_samples_per_bin =  len(angles)/bin_count
hist, bins = np.histogram(angles, bin_count)

delete_indices = []
for i in range(len(angles)):
    for j in range(bin_count):
        if angles[i] > bins[j] and angles[i] <= bins[j+1] and np.random.rand() > avg_samples_per_bin/hist[j]:
                delete_indices.append(i)
image_path = np.delete(image_path, delete_indices, axis=0)
angles = np.delete(angles, delete_indices)


X_path_train, X_paths_test, Y_train, Y_test = train_test_split(image_path, angles, test_size=0.1)
print('Train:', X_path_train.shape, Y_train.shape)
print('Test:', X_paths_test.shape, Y_test.shape)




# generators for train and validation data
print ("X_path_train ",X_path_train)
train_gen = get_data(X_path_train, Y_train)
val_gen = get_data(X_path_train, Y_train, validation=True)

model = get_model()
#train network
history = model.fit_generator(train_gen, validation_data=val_gen, nb_val_samples=len(Y_train), samples_per_epoch=len(Y_train), 
                              nb_epoch=1, verbose=2 )

print(model.summary())

# Save model data
model.save('./model.h5')

