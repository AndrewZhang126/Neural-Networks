# The following code was adapted from Week 1 Programming Assignment 1 in the Improving Deep Neural Networks course by DeepLearning.AI offered on Coursera
# https://www.coursera.org/learn/deep-neural-network/home/week/1


import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')



# load dataset
train_X, train_Y, test_X, test_Y = load_dataset()



"""
This function implements a 3-layer neural network [LINEAR->RELU] -> [LINEAR->RELU] -> [LINEAR->SIGMOID]
"""
def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    
    grads = {}
    costs = [] 
    m = X.shape[1] 
    layers_dims = [X.shape[0], 10, 5, 1]
    
    # initialize parameters 
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    for i in range(num_iterations):

        # forward propagation
        a3, cache = forward_propagation(X, parameters)
        
        # compute loss
        cost = compute_loss(a3, Y)

        # backward propagation
        grads = backward_propagation(X, Y, cache)
        
        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters



"""
This function implements zero initialization
"""
def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)            
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters



"""
This function implements random initialization
"""
def initialize_parameters_random(layers_dims):
    parameters = {}
    L = len(layers_dims)            
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters



"""
This function implements He initialization
"""
def initialize_parameters_he(layers_dims):
    parameters = {}
    L = len(layers_dims) - 1 
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters



# initialization value can either be "zeros", "random", or "he"
parameters = model(train_X, train_Y, initialization = "zeros")

print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)


# # References
# [1] https://www.coursera.org/learn/deep-neural-network/programming/QF47Q/initialization
