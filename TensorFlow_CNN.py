#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[36]:





# In[37]:


happy_model = happyModel()
# Print a summary for each layer
for layer in summary(happy_model):
    print(layer)
    
output = [['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
            ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform'],
            ['BatchNormalization', (None, 64, 64, 32), 128],
            ['ReLU', (None, 64, 64, 32), 0],
            ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid'],
            ['Flatten', (None, 32768), 0],
            ['Dense', (None, 1), 32769, 'sigmoid']]
    
comparator(summary(happy_model), output)


# In[42]:


# load the data
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()


# In[44]:


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


# In[57]:


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


# In[58]:


conv_model = convolutional_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()
    
output = [['InputLayer', [(None, 64, 64, 3)], 0],
        ['Conv2D', (None, 64, 64, 8), 392, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 64, 64, 8), 0],
        ['MaxPooling2D', (None, 8, 8, 8), 0, (8, 8), (8, 8), 'same'],
        ['Conv2D', (None, 8, 8, 16), 528, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 8, 8, 16), 0],
        ['MaxPooling2D', (None, 2, 2, 16), 0, (4, 4), (4, 4), 'same'],
        ['Flatten', (None, 64), 0],
        ['Dense', (None, 6), 390, 'softmax']]
    
comparator(summary(conv_model), output)


# In[59]:


# train the model
rain_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)


# # References
# [1] https://www.coursera.org/learn/convolutional-neural-networks/programming/7Bfm2/convolution-model-application
# [2] https://www.tensorflow.org/guide/keras/sequential_model
# [3] https://www.tensorflow.org/guide/keras/functional
