# The following code was adapted from Week 2 Programming Assignment 2 in the Convolutional Neural Networks course by DeepLearning.AI offered on Coursera
# https://www.coursera.org/learn/convolutional-neural-networks/home/week/2



import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation



# create and split dataset
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
directory = "dataset/"
train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)
validation_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='validation',
                                             seed=42)



AUTOTUNE = tf.data.experimental.AUTOTUNE

#prevents memory bottleneck when reading from disk
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)



"""
This function implements data augmentation through flipping and rotation
"""
def data_augmenter():
    
    data_augmentation = tf.keras.Sequential([
        tfl.experimental.preprocessing.RandomFlip('horizontal'),
        tfl.experimental.preprocessing.RandomRotation(0.2)
    ])
    
    return data_augmentation



data_augmentation = data_augmenter()

# display augmented data
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')

        
        
# References
# [1] https://www.coursera.org/learn/convolutional-neural-networks/programming/nZima/transfer-learning-with-mobilenet
