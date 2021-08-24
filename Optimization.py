# The following code was adapted from Week 2 Programming Assignment 1 in the Improving Deep Neural Networks course by DeepLearning.AI offered on Coursera
# https://www.coursera.org/learn/deep-neural-network/home/week/2



import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (7.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')



"""
This function implements one step of gradient descent
"""

def update_parameters_with_gd(parameters, grads, learning_rate):
   
    L = len(parameters) // 2 

    # perform update rule for each parameter
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - (learning_rate * grads['dW' + str(l)])
        parameters["b" + str(l)] = parameters["b" + str(l)] - (learning_rate * grads['db' + str(l)])
        
    return parameters



"""
This function creates random minibatches
"""
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    m = X.shape[1]                 
    mini_batches = []
        
    # shuffle training examples
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    
    inc = mini_batch_size

    # partition shuffled examples
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * inc : (k + 1) * inc]
        mini_batch_Y = shuffled_Y[:, k * inc : (k + 1) * inc]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * inc:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * inc:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches



"""
This function implements velocity
"""
def initialize_velocity(parameters):
    
    L = len(parameters) // 2 
    v = {}
    
    for l in range(1, L + 1):
        
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        
    return v



"""
This function updates parameters using momentum
"""

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    
    L = len(parameters) // 2 
    
    # momentum update for each parameter
    for l in range(1, L + 1):

        v["dW" + str(l)] = (beta * v["dW" + str(l)]) + ((1 - beta) * grads['dW' + str(l)])
        v["db" + str(l)] = (beta * v["db" + str(l)]) + ((1 - beta) * grads['db' + str(l)])
        parameters["W" + str(l)] = parameters["W" + str(l)] - (learning_rate * v["dW" + str(l)])
        parameters["b" + str(l)] = parameters["b" + str(l)] - (learning_rate * v["db" + str(l)])
        
    return parameters, v



"""
This function initializes Adam variables
"""

def initialize_adam(parameters) :
 
    L = len(parameters) // 2 
    v = {}
    s = {}
    
    for l in range(1, L + 1):

        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
    
    return v, s



"""
This function updates parameters using Adam
"""

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):

    
    L = len(parameters) // 2                
    v_corrected = {}                         
    s_corrected = {}                        
    
    # perform Adam update on all parameters
    for l in range(1, L + 1):
        
        v["dW" + str(l)] = (beta1 * v["dW" + str(l)]) + ((1 - beta1) * grads['dW' + str(l)])
        v["db" + str(l)] = (beta1 * v["db" + str(l)]) + ((1 - beta1) * grads['db' + str(l)])

        # compute bias-corrected first moment estimate
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)

        # compute moving average of the squared gradients
        s["dW" + str(l)] = (beta2 * s["dW" + str(l)]) + ((1 - beta2) * np.power(grads['dW' + str(l)], 2))
        s["db" + str(l)] = (beta2 * s["db" + str(l)]) + ((1 - beta2) * np.power(grads['db' + str(l)], 2))

        # compute bias-corrected second raw moment estimate
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)
        
        # update parameters
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * (v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon))
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * (v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon))

    return parameters, v, s, v_corrected, s_corrected



# load the dataset
train_X, train_Y = load_dataset()



"""
This function implements a 3-layer neural network
"""
def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 5000, print_cost = True):

    L = len(layers_dims)             
    costs = []                       
    t = 0     
    seed = 10    
    m = X.shape[1]                  
    
    # initialize parameters
    parameters = initialize_parameters(layers_dims)

    # initialize the optimizer
    if optimizer == "gd":
        pass 
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    # optimization loop
    for i in range(num_epochs):
        
        # define the random minibatches
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        
        for minibatch in minibatches:

            # select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)

            # compute loss
            cost_total += compute_cost(a3, minibatch_Y)

            # backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 
                parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)
        cost_avg = cost_total / m
        
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)
                
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters



# train 3-layer model with gradient descent
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")

# make predictions
predictions = predict(train_X, train_Y, parameters)

# plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)



# train 3-layer model with momentum
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")

# make predictions
predictions = predict(train_X, train_Y, parameters)

# plot decision boundary
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)



# train 3-layer model with Adam
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")

# make predictions
predictions = predict(train_X, train_Y, parameters)

# plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)



"""
This function implements a 3-layer neurla network with learning rate decay
"""
def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 5000, print_cost = True, decay=None, decay_rate=1):

    L = len(layers_dims)            
    costs = []                       
    t = 0                           
    seed = 10                        
    m = X.shape[1]                   
    lr_rates = []
    learning_rate0 = learning_rate   
    
    # initialize parameters
    parameters = initialize_parameters(layers_dims)

    # initialize the optimizer
    if optimizer == "gd":
        pass 
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    # optimization loop
    for i in range(num_epochs):
        
        # define the random minibatches
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        
        for minibatch in minibatches:

            # select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)

            # compute loss
            cost_total += compute_cost(a3, minibatch_Y)

            # backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 
                parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)
        cost_avg = cost_total / m
        if decay:
            learning_rate = decay(learning_rate0, i, decay_rate)
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost_avg))
            if decay:
                print("learning rate after epoch %i: %f"%(i, learning_rate))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)
                
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters



"""
This function updates the learning rate using exponential weight decay
"""
def update_lr(learning_rate0, epoch_num, decay_rate):

    learning_rate = (1 / (1 + decay_rate * epoch_num)) * learning_rate0
    
    return learning_rate

   

# train 3-layer model with gradient descent and exponential learning rate decay
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "gd", learning_rate = 0.1, num_epochs=5000, decay=update_lr)

# make predictions
predictions = predict(train_X, train_Y, parameters)

# plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)



"""
This function updates the learning rate using exponential weight decay with fixed interval scheduling
"""
def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):

    learning_rate = (1 / (1 + decay_rate * np.floor(epoch_num / time_interval))) * learning_rate0
    
    return learning_rate



# train 3-layer model with gradient descent and learning rate scheduling
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "gd", learning_rate = 0.1, num_epochs=5000, decay=schedule_lr_decay)

# make predictions
predictions = predict(train_X, train_Y, parameters)

# plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)



# train 3-layer model with momentum and learning rate scheduling
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "momentum", learning_rate = 0.1, num_epochs=5000, decay=schedule_lr_decay)

# make predictions
predictions = predict(train_X, train_Y, parameters)

# plot decision boundary
plt.title("Model with Gradient Descent with momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)



# train 3-layer model with adam and learning rate scheduling
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "adam", learning_rate = 0.01, num_epochs=5000, decay=schedule_lr_decay)

# make predictions
predictions = predict(train_X, train_Y, parameters)

# plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)



# # References
# [1] https://www.coursera.org/learn/deep-neural-network/programming/390Oe/optimization-methods
