#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


parameters, grads, learning_rate = update_parameters_with_gd_test_case()
learning_rate = 0.01
parameters = update_parameters_with_gd(parameters, grads, learning_rate)

print("W1 =\n" + str(parameters["W1"]))
print("b1 =\n" + str(parameters["b1"]))
print("W2 =\n" + str(parameters["W2"]))
print("b2 =\n" + str(parameters["b2"]))

update_parameters_with_gd_test(update_parameters_with_gd)


# In[21]:


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


# In[23]:


t_X, t_Y, mini_batch_size = random_mini_batches_test_case()
mini_batches = random_mini_batches(t_X, t_Y, mini_batch_size)

print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

random_mini_batches_test(random_mini_batches)


# In[26]:


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


# In[27]:


parameters = initialize_velocity_test_case()

v = initialize_velocity(parameters)
print("v[\"dW1\"] =\n" + str(v["dW1"]))
print("v[\"db1\"] =\n" + str(v["db1"]))
print("v[\"dW2\"] =\n" + str(v["dW2"]))
print("v[\"db2\"] =\n" + str(v["db2"]))

initialize_velocity_test(initialize_velocity)


# In[30]:


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


# In[31]:


parameters, grads, v = update_parameters_with_momentum_test_case()

parameters, v = update_parameters_with_momentum(parameters, grads, v, beta = 0.9, learning_rate = 0.01)
print("W1 = \n" + str(parameters["W1"]))
print("b1 = \n" + str(parameters["b1"]))
print("W2 = \n" + str(parameters["W2"]))
print("b2 = \n" + str(parameters["b2"]))
print("v[\"dW1\"] = \n" + str(v["dW1"]))
print("v[\"db1\"] = \n" + str(v["db1"]))
print("v[\"dW2\"] = \n" + str(v["dW2"]))
print("v[\"db2\"] = v" + str(v["db2"]))

update_parameters_with_momentum_test(update_parameters_with_momentum)


# <a name='ex-5'></a>   
# ### Exercise 5 - initialize_adam
# 
# Initialize the Adam variables $v, s$ which keep track of the past information.
# 
# **Instruction**: The variables $v, s$ are python dictionaries that need to be initialized with arrays of zeros. Their keys are the same as for `grads`, that is:
# for $l = 1, ..., L$:
# ```python
# v["dW" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["W" + str(l)])
# v["db" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["b" + str(l)])
# s["dW" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["W" + str(l)])
# s["db" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["b" + str(l)])
# 
# ```

# In[34]:


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


# In[35]:


parameters = initialize_adam_test_case()

v, s = initialize_adam(parameters)
print("v[\"dW1\"] = \n" + str(v["dW1"]))
print("v[\"db1\"] = \n" + str(v["db1"]))
print("v[\"dW2\"] = \n" + str(v["dW2"]))
print("v[\"db2\"] = \n" + str(v["db2"]))
print("s[\"dW1\"] = \n" + str(s["dW1"]))
print("s[\"db1\"] = \n" + str(s["db1"]))
print("s[\"dW2\"] = \n" + str(s["dW2"]))
print("s[\"db2\"] = \n" + str(s["db2"]))

initialize_adam_test(initialize_adam)


# In[46]:


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


# In[47]:


parametersi, grads, vi, si = update_parameters_with_adam_test_case()

t = 2
learning_rate = 0.02
beta1 = 0.8
beta2 = 0.888
epsilon = 1e-2

parameters, v, s, vc, sc  = update_parameters_with_adam(parametersi, grads, vi, si, t, learning_rate, beta1, beta2, epsilon)
print(f"W1 = \n{parameters['W1']}")
print(f"W2 = \n{parameters['W2']}")
print(f"b1 = \n{parameters['b1']}")
print(f"b2 = \n{parameters['b2']}")

update_parameters_with_adam_test(update_parameters_with_adam)


# In[ ]:


# load the dataset
train_X, train_Y = load_dataset()


# In[49]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[53]:


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


# In[54]:


"""
This function updates the learning rate using exponential weight decay
"""
def update_lr(learning_rate0, epoch_num, decay_rate):

    learning_rate = (1 / (1 + decay_rate * epoch_num)) * learning_rate0
    
    return learning_rate


# In[55]:


learning_rate = 0.5
print("Original learning rate: ", learning_rate)
epoch_num = 2
decay_rate = 1
learning_rate_2 = update_lr(learning_rate, epoch_num, decay_rate)

print("Updated learning rate: ", learning_rate_2)

update_lr_test(update_lr)


# In[ ]:


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


# In[65]:


"""
This function updates the learning rate using exponential weight decay with fixed interval scheduling
"""

def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):

    learning_rate = (1 / (1 + decay_rate * np.floor(epoch_num / time_interval))) * learning_rate0
    
    return learning_rate


# In[66]:


learning_rate = 0.5
print("Original learning rate: ", learning_rate)

epoch_num_1 = 10
epoch_num_2 = 100
decay_rate = 0.3
time_interval = 100
learning_rate_1 = schedule_lr_decay(learning_rate, epoch_num_1, decay_rate, time_interval)
learning_rate_2 = schedule_lr_decay(learning_rate, epoch_num_2, decay_rate, time_interval)
print("Updated learning rate after {} epochs: ".format(epoch_num_1), learning_rate_1)
print("Updated learning rate after {} epochs: ".format(epoch_num_2), learning_rate_2)

schedule_lr_decay_test(schedule_lr_decay)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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
