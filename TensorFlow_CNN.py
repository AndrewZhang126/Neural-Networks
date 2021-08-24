# The following code was adapted from Week 1 Programming Assignment 2 in the Convolutional Neural Networks course by DeepLearning.AI offered on Coursera
# https://www.coursera.org/learn/convolutional-neural-networks/home/week/1



import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops

get_ipython().run_line_magic('matplotlib', 'inline')



# load the data
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()



"""
This function performs a one hot conversion
"""
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y



# split data into train/test sets
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T



"""
This function imeplements forward propagation for a CONV -> RELU -> MAXPOOL -> CONV -> RELU -> MAXPOOl -> FC CNN model
"""
def convolutional_model(input_shape):

    input_img = tf.keras.Input(shape=input_shape)
    
    # Conv layer with 8 4x4 filters
    Z1 = tf.keras.layers.Conv2D(filters=8, kernel_size=4, padding='same')(input_img)
    # Relu activation
    A1 = tf.keras.layers.ReLU()(Z1)
    
    # MaxPool layer
    P1 = tf.keras.layers.MaxPool2D(pool_size=(8, 8), strides=(8, 8), padding='same')(A1)
    
    # Conv layer with 16 2x2 filters
    Z2 = tf.keras.layers.Conv2D(filters=16, kernel_size=2, padding='same')(P1)
    
    # Relu activation
    A2 = tf.keras.layers.ReLU()(Z2)
    
    # MaxPool layer
    P2 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='same')(A2)
    
    # Flatten allows previous layer to connect to a fully connected layer
    F = tf.keras.layers.Flatten()(P2)
    
    # Fully connected (Dense) layer
    outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    
    return model



# train the model
rain_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)



# References
# [1] https://www.coursera.org/learn/convolutional-neural-networks/programming/7Bfm2/convolution-model-application
# [2] https://www.tensorflow.org/guide/keras/sequential_model
# [3] https://www.tensorflow.org/guide/keras/functional
