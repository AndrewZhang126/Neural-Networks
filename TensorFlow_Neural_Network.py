# The following code was adapted from Week 3 Programming Assignment 1 in the Improving Deep Neural Networks course by DeepLearning.AI offered on Coursera
# https://www.coursera.org/learn/deep-neural-network/home/week/3



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import time



# you can specify your own dataset for training
train_dataset = folder.File('...', "r")
test_dataset = folder.File('...', "r")
x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])
x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])



"""
This function transforms and normalizes an image into a tensor
"""
def normalize(image):
    image = tf.cast(image, tf.float32) / 256.0
    image = tf.reshape(image, [-1,1])
    return image



# normalize data
new_train = x_train.map(normalize)
new_test = x_test.map(normalize)



"""
This function implements the linear function Y = WX + b
"""
def linear_function():
    
    # initalize variables
    X = tf.constant(np.random.randn(3, 1), name = 'X')
    W = tf.constant(np.random.randn(4, 3), name = 'W')
    b = tf.constant(np.random.randn(4, 1), name = 'b')
    Y = tf.add(tf.matmul(W, X), b)

    return Y



"""
This function implements the sigmoid function
"""
def sigmoid(z):
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)
    return a



"""
This function implements one hot encoding for a single label
"""
def one_hot_matrix(label, depth=6):
    one_hot = tf.reshape(tf.one_hot(label, depth, axis=0), (depth, 1))
    return one_hot



"""
This function initializes parameters using TensorFlow
"""
def initialize_parameters():
             
    # draws samples from a distribution 
    initializer = tf.keras.initializers.GlorotNormal()   
    
    # stores the state of each variable with the correct shape specific to this neural network
    W1 = tf.Variable(initializer(shape=(25, 12288)))
    b1 = tf.Variable(initializer(shape=(25, 1)))
    W2 = tf.Variable(initializer(shape=(12, 25)))
    b2 = tf.Variable(initializer(shape=(12, 1)))
    W3 = tf.Variable(initializer(shape=(6, 12)))
    b3 = tf.Variable(initializer(shape=(6, 1)))

    # stores parameter values
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters



"""
This function implements forward propagation using TensorFlow and Keras
"""
# builds a computational graph, which will keep track of operations to calculate gradients for backward propagation
@tf.function

def forward_propagation(X, parameters):
    
    # X is the input data
    
    # retrieve the parameters
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    # perform the [linear -> relu] layers
    Z1 = tf.math.add(tf.linalg.matmul(W1, X), b1)
    A1 = tf.keras.activations.relu(Z1)
    Z2 = tf.math.add(tf.linalg.matmul(W2, A1), b2)
    A2 = tf.keras.activations.relu(Z2)
    Z3 = tf.math.add(tf.linalg.matmul(W3, A2), b3)
    
    return Z3



"""
This function computes the cross entropy cost function
"""
@tf.function
def compute_cost(logits, labels):

    # logits is the output of forward propagation
    # labels are the true label values
    
    # calculates the cost function
    cost = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true = labels, y_pred = logits, from_logits=True))
    
    return cost



"""
This function constructs a 3-layer [linear -> relu] neural network using TensorFlow and Keras
"""
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    
    # X_train is the training set with 1080 examples
    # Y_train is the test set with 1080 examples
    # X_test is the training set with 120 examples
    # Y_test is the test set with 120 examples
    # learning_rate is the learing rate for optimization
    # num_epochs is the number of epochs for optimization
    # minibatch_size is the size of a minibatch
    # print_cost means whether or not to print the cost 
    
    # keep track of the cost
    costs = []                                        
    
    # initialize parameters
    parameters = initialize_parameters()

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    # uses the SDG optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    # prevent memory bottleneck
    X_train = X_train.batch(minibatch_size, drop_remainder=True).prefetch(8)    
    Y_train = Y_train.batch(minibatch_size, drop_remainder=True).prefetch(8)

    # perform training
    for epoch in range(num_epochs):

        epoch_cost = 0.
        
        # select a minibatch
        for (minibatch_X, minibatch_Y) in zip(X_train, Y_train):
            
            # GradientTape records operations for backward propagation
            with tf.GradientTape() as tape:
                
                # forward propagation
                Z3 = forward_propagation(minibatch_X, parameters)
                
                # compute cost
                minibatch_cost = compute_cost(Z3, minibatch_Y)
              
            # backward propagation and update parameters
            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(minibatch_cost, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_cost += minibatch_cost / minibatch_size

        # print the cost after every epoch
        if print_cost == True and epoch % 10 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 5 == 0:
            costs.append(epoch_cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters



# References
# [1] https://www.coursera.org/learn/deep-neural-network/programming/fuJJY/tensorflow-introduction
# [2] https://www.tensorflow.org/guide/autodiff 
# [3] https://www.tensorflow.org/api_docs/python/tf/GradientTape
