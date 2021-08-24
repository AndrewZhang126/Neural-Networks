#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import time


# In[3]:


# you can specify your own dataset for training
train_dataset = folder.File('...', "r")
test_dataset = folder.File('...', "r")
x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])
x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])


# There's one more additional difference between TensorFlow datasets and Numpy arrays: If you need to transform one, you would invoke the `map` method to apply the function passed as an argument to each of the elements.

# In[9]:


"""
This function transforms and normalizes an image into a tensor
"""
def normalize(image):
    image = tf.cast(image, tf.float32) / 256.0
    image = tf.reshape(image, [-1,1])
    return image


# In[10]:


# normalize data
new_train = x_train.map(normalize)
new_test = x_test.map(normalize)


# In[15]:


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


# In[16]:


result = linear_function()
print(result)

assert type(result) == EagerTensor, "Use the TensorFlow API"
assert np.allclose(result, [[-2.15657382], [ 2.95891446], [-1.08926781], [-0.84538042]]), "Error"
print("\033[92mAll test passed")


# In[17]:


"""
This function implements the sigmoid function
"""
def sigmoid(z):
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)
    return a


# In[18]:


result = sigmoid(-1)
print ("type: " + str(type(result)))
print ("dtype: " + str(result.dtype))
print ("sigmoid(-1) = " + str(result))
print ("sigmoid(0) = " + str(sigmoid(0.0)))
print ("sigmoid(12) = " + str(sigmoid(12)))

def sigmoid_test(target):
    result = target(0)
    assert(type(result) == EagerTensor)
    assert (result.dtype == tf.float32)
    assert sigmoid(0) == 0.5, "Error"
    assert sigmoid(-1) == 0.26894143, "Error"
    assert sigmoid(12) == 0.9999939, "Error"

    print("\033[92mAll test passed")

sigmoid_test(sigmoid)


# <a name='2-3'></a>
# ### 2.3 - Using One Hot Encodings
# 
# Many times in deep learning you will have a $Y$ vector with numbers ranging from $0$ to $C-1$, where $C$ is the number of classes. If $C$ is for example 4, then you might have the following y vector which you will need to convert like this:
# 
# 
# <img src="images/onehot.png" style="width:600px;height:150px;">
# 
# This is called "one hot" encoding, because in the converted representation, exactly one element of each column is "hot" (meaning set to 1). To do this conversion in numpy, you might have to write a few lines of code. In TensorFlow, you can use one line of code: 
# 
# - [tf.one_hot(labels, depth, axis=0)](https://www.tensorflow.org/api_docs/python/tf/one_hot)
# 
# `axis=0` indicates the new axis is created at dimension 0
# 
# <a name='ex-3'></a>
# ### Exercise 3 - one_hot_matrix
# 
# Implement the function below to take one label and the total number of classes $C$, and return the one hot encoding in a column wise matrix. Use `tf.one_hot()` to do this, and `tf.reshape()` to reshape your one hot tensor! 
# 
# - `tf.reshape(tensor, shape)`

# In[37]:


"""
This function implements one hot encoding for a single label
"""
def one_hot_matrix(label, depth=6):
    one_hot = tf.reshape(tf.one_hot(label, depth, axis=0), (depth, 1))
    return one_hot


# In[38]:


def one_hot_matrix_test(target):
    label = tf.constant(1)
    depth = 4
    result = target(label, depth)
    print(result)
    assert result.shape[0] == depth, "Use the parameter depth"
    assert result.shape[1] == 1, f"Reshape to have only 1 column"
    assert np.allclose(result,  [[0.], [1.], [0.], [0.]] ), "Wrong output. Use tf.one_hot"
    result = target(3, depth)
    assert np.allclose(result, [[0.], [0.], [0.], [1.]] ), "Wrong output. Use tf.one_hot"
    
    print("\033[92mAll test passed")

one_hot_matrix_test(one_hot_matrix)


# In[43]:


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


# In[44]:


def initialize_parameters_test(target):
    parameters = target()

    values = {"W1": (25, 12288),
              "b1": (25, 1),
              "W2": (12, 25),
              "b2": (12, 1),
              "W3": (6, 12),
              "b3": (6, 1)}

    for key in parameters:
        print(f"{key} shape: {tuple(parameters[key].shape)}")
        assert type(parameters[key]) == ResourceVariable, "All parameter must be created using tf.Variable"
        assert tuple(parameters[key].shape) == values[key], f"{key}: wrong shape"
        assert np.abs(np.mean(parameters[key].numpy())) < 0.5,  f"{key}: Use the GlorotNormal initializer"
        assert np.std(parameters[key].numpy()) > 0 and np.std(parameters[key].numpy()) < 1, f"{key}: Use the GlorotNormal initializer"

    print("\033[92mAll test passed")
    
initialize_parameters_test(initialize_parameters)


# In[46]:


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


# In[47]:


def forward_propagation_test(target, examples):
    for batch in examples:
        forward_pass = target(batch, parameters)
        assert type(forward_pass) == EagerTensor, "Your output is not a tensor"
        assert forward_pass.shape == (6, 1), "Last layer must use W3 and b3"
        assert np.any(forward_pass < 0), "Don't use a ReLu layer at end of your network"
        assert np.allclose(forward_pass, 
                           [[-0.13082162],
                           [ 0.21228778],
                           [ 0.7050022 ],
                           [-1.1224034 ],
                           [-0.20386729],
                           [ 0.9526217 ]]), "Output does not match"
        print(forward_pass)
        break
    

    print("\033[92mAll test passed")

forward_propagation_test(forward_propagation, new_train)


# In[48]:


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


# In[49]:


def compute_cost_test(target):
    labels = np.array([[0., 1.], [0., 0.], [1., 0.]])
    logits = np.array([[0.6, 0.4], [0.4, 0.6], [0.4, 0.6]])
    result = compute_cost(logits, labels)
    print(result)
    assert(type(result) == EagerTensor), "Use the TensorFlow API"
    assert (np.abs(result - (0.7752516 +  0.9752516 + 0.7752516) / 3.0) < 1e-7), "Test does not match. Did you get the mean of your cost functions?"

    print("\033[92mAll test passed")

compute_cost_test(compute_cost)


# In[50]:


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


# # References
# [1] https://www.coursera.org/learn/deep-neural-network/programming/fuJJY/tensorflow-introduction
# [2] https://www.tensorflow.org/guide/autodiff 
# [3] https://www.tensorflow.org/api_docs/python/tf/GradientTape
