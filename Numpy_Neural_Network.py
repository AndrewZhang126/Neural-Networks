# The following code was adapted from Week 4 Programming Assignments 1 and 2 in the Neural Networks and Deep Learning course by DeepLearning.AI offered on Coursera
# https://www.coursera.org/learn/neural-networks-deep-learning/home/week/4



import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')



"""
This function initalizes the parameters in a deep L-layer neural network
"""
def initialize_parameters_deep(layer_dims):
    
    # dictionary containing parameters from W1, b1 to WL, bL
    parameters = {}
    
    # number of layers in the network
    L = len(layer_dims) 

    
    for l in range(1, L): 
        # initalize parameters for each layer of the neural network
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        # ensure the parameters have the right shape
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1])) 
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters



"""
This function implements a sigmoid activation
"""
def sigmoid(Z):
    
    A = 1/(1+np.exp(-Z))
    
    # cache helps with backward propagation
    cache = Z
    
    return A, cache



"""
This function implements a RELU activation
"""
def relu(Z):
    
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    
    # cache helps with backward propagation
    cache = Z

    return A, cache



"""
This function implements the linear calculation for a single layer's forward propagation
"""
def linear_forward(A, W, b):
    
    # A is the activations from the previous layer
    # W is the weights parameter
    #b is the bias parameter
    
    # equation for linear calculation is (W*A) + b
    Z = np.dot(W, A) + b
    
    # cache is a tuple containing W, A and b, which are needed for backward propagation 
    cache = (A, W, b)
    
    return Z, cache



"""
This function implements the activation function for a single layer's forward propagation
"""
def linear_activation_forward(A_prev, W, b, activation):
    
    # A_prev is the activation from the previous layer
    # activation is a string specifying the type of activation 
    
    if activation == "sigmoid":
        
        # applies linear calcuation for forward propagation
        Z, linear_cache = linear_forward(A_prev, W, b)
        
        # applies activation function
        A, activation_cache = sigmoid(Z)
        
        # YOUR CODE ENDS HERE
    
    elif activation == "relu":
        
        # applies linear calcuation for forward propagation
        Z, linear_cache = linear_forward(A_prev, W, b)
        
        # applies activation function
        A, activation_cache = relu(Z)
        
    
    # cache stores values for backward propagation
    cache = (linear_cache, activation_cache)

    # A will be the activation value, or the output for this layer to be passed to the next layer
    return A, cache



"""
This function implements forward propagation for an L-layer neural network
"""
def L_model_forward(X, parameters):
    
    # X is the input data
    # parameters is the output of the initialize_parameters_deep() function
    
    # stores values for backward propagation
    caches = []
    
    # first activation is the input data
    A = X
    
    # number of layers in the neural network
    L = len(parameters) // 2                 
    
    # implements forward propagation for the [linear -> relu] layers
    for l in range(1, L):
        A_prev = A 
        
        # helper function performs forward porpagation for activation
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b'+ str(l)], 'relu')
        caches.append(cache)
        
        
    # implements forard propagation for the [linear -> sigmoid] layer
    # helper function performs forward porpagation for activation
    AL, cache = linear_activation_forward(A, parameters['W'+ str(L)], parameters['b'+ str(L)], 'sigmoid')
    caches.append(cache)
          
    # AL is the last activation value
    return AL, caches



"""
This function computes the cross-entropy cost
"""

def compute_cost(AL, Y):
    
    # AL is the prediction label value of the neural network
    # Y is the true label value
    
    # contains the number of training inputs
    m = Y.shape[1]

    # computes the cost according to the cost function
    cost = (-1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)))   
    
   # ensures the shape of the cost is correct
    cost = np.squeeze(cost)     
    
    return cost



"""
This function implements backward propagation for a sigmoid unit
"""
def sigmoid_backward(dA, cache):
    
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    
    return dZ



"""
This function implements backward propagation for a RELU unit
"""
def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True)
    
    # when z <= 0, dz should be 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ



"""
This function implements the linear calculation for a single layer's backward propagation
"""
def linear_backward(dZ, cache):
    
    # dZ is the gradient of the cost with respect to the linear output of the current layer
    # cache contains A_prev, W, and b from the forward propagation of the current layer
    
    A_prev, W, b = cache
    m = A_prev.shape[1]

    # calculates dW, db, and dA_prev from their respective equations
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db



"""
This function implements the activation function for a single layer's backward propagation
"""
def linear_activation_backward(dA, cache, activation):
    
    # dA is activation gradient for the current layer
    # activation is a string specifying the type of activation 
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
         
        # implements the backward propagation for relu
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        
        # implements the backward propagation for sigmoid
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db



"""
This function implements forward propagation for an L-layer neural network
"""
def L_model_backward(AL, Y, caches):
    
    # AL is the output of L_model_forward(), which is the forward propagation
    # Y is the true label value
    
    # dictionary with gradient values
    grads = {}
    
    # the number of layers
    L = len(caches) 
    
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    
    # initialize backward propagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) 
    
    
    # implements backward propagation for the [linear -> sigmoid] layer
    current_cache = caches[L-1]
    
    # helper function performs backward propagation for activation
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, 'sigmoid')
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
   
    
    # implements backward propagation for the [linear -> relu] layers
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        
        # helper function performs backward propagation for activation
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads



"""
This function updatea parameters using gradient descent
"""
def update_parameters(params, grads, learning_rate):
  
    # params contains parameter values
    # grads contains gradient values, which are outputs from L_model_backward()
    
    parameters = params.copy()
    
    # number of layers in the neural network
    L = len(parameters) // 2 

    # update parameters for each layer according to the equation
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l + 1)] - (learning_rate * grads["dW" + str(l + 1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l + 1)] - (learning_rate * grads["db" + str(l + 1)])
        
    return parameters



"""
This function implements a L-layer neural network of the form [linear->relu]*(L-1)->[linear->sigmoid]  
"""
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
   
    # X is the input data
    # Y is the true label value
    # layers_dim is a list containing the input and layer size for each layer
    # learning_rate is the learning rate for gradient descent update
    # num_iterations is the number of iterations for optimization
    # print_cost means whether or not to print out the cost
    
    # keep track of cost
    costs = []                         
    
    # initalize parameters
    parameters = initialize_parameters_deep(layers_dims)
    
    # perform gradient descent
    for i in range(0, num_iterations):

        # perform forward propagation
        AL, caches = L_model_forward(X, parameters)
        
        # compute cost
        cost = compute_cost(AL, Y)
        
        # perform backward propagation
        grads = L_model_backward(AL, Y, caches)
        
        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    
    return parameters, costs



#  4-layer model
layers_dims = [12288, 20, 7, 5, 1]

# train model and print out cost
parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

# train accuracy
pred_train = predict(train_x, train_y, parameters)

# test accuracy
pred_test = predict(test_x, test_y, parameters)



# display images that were labeled incorrectly
print_mislabeled_images(classes, test_x, test_y, pred_test)



# References
# [1] https://www.coursera.org/learn/neural-networks-deep-learning/programming/GY8CB/building-your-deep-neural-network-step-by-step/lab
# [2] https://www.coursera.org/learn/neural-networks-deep-learning/programming/Sfu8g/deep-neural-network-application/lab
