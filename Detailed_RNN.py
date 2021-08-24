# The following code was adapted from Week 1 Programming Assignment 1 in the Sequence Models course by DeepLearning.AI offered on Coursera
# https://www.coursera.org/learn/nlp-sequence-models/home/week/1



import numpy as np



# softmax activation function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)



# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))



"""
This function implements the forward step of a single RNN cell
"""
def rnn_cell_forward(xt, a_prev, parameters):
    
    # retrieve parameters
    # Wax is the weight value for the current input
    Wax = parameters["Wax"]
    # Waa is the weight value for the previous hidden state 
    Waa = parameters["Waa"]
    # Waa is the weight value for the current hidden state's output
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    # apply the first activation function (tanh)
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    # apply the second activation function (softmax) on the output
    yt_pred = softmax(np.dot(Wya, a_next) + by)
    
    # cache stores values for backward propagation
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache



"""
This function implements forward propagation for a RNN
"""
def rnn_forward(x, a0, parameters):
    
    # x is the input data for every time step
    # a0 is the inital hidden state
    
    caches = []
    
    # retrieve dimensions
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
   
    # initialize a and y_pred with zeros 
    a = np.zeros((n_a, m, T_x))
    
    # y_pred is the prediction
    y_pred = np.zeros((n_y, m, T_x))
    
    a_next = a0
    
    # loop over all time steps
    for t in range(T_x):
        
        # perform calculations for the current RNN cell
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        
        # save the values of the next hidden state, prediction, and cache
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)
    
    # cache stores values needed for backward propagation
    caches = (caches, x)
    
    return a, y_pred, caches



"""
This function implements the forward step of a single LSTM cell
"""
def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    
    # xt is the input data for this time step
    # a_prev is the previous hidden state

    # retrieve parameters 
    Wf = parameters["Wf"] 
    bf = parameters["bf"]
    Wi = parameters["Wi"] 
    bi = parameters["bi"] 
    Wc = parameters["Wc"] 
    bc = parameters["bc"]
    Wo = parameters["Wo"] 
    bo = parameters["bo"]
    Wy = parameters["Wy"] 
    by = parameters["by"]
    
    # retrieve dimensions 
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # concatenate a_prev and xt
    concat = np.concatenate((a_prev, xt))

    # compute forget gate
    ft = sigmoid(np.dot(Wf, concat) + bf)
    
    # compute update gate
    it = sigmoid(np.dot(Wi, concat) + bi)
    
    # compute candidate value
    cct = np.tanh(np.dot(Wc, concat) + bc)
    
    # compute cell state
    c_next = (ft * c_prev) + (it * cct)
    
    # compute output gate
    ot = sigmoid(np.dot(Wo, concat) + bo)
    
    # compute hidden state
    a_next = ot * np.tanh(c_next)
    
    # compute prediction
    yt_pred = softmax(np.dot(Wy, a_next) + by)

    # cache stores values needed for backward propagation
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache



"""
This function implements forward propagation for an RNN using LSTM cells
"""
def lstm_forward(x, a0, parameters):
   
    # x is the input data for every time step
    # a0 is the initial hidden state

    caches = []
    
    Wy = parameters['Wy'] 
    
    # retrieve dimensions
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape
    
    # initialize "a", "c" and "y" with zeros
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    # initialize a_next and c_next
    a_next = a0
    c_next = np.zeros((n_a, m))
    
    # loop over all time-steps
    for t in range(T_x):
        
        # get the current time step input data
        xt = x[:,:,t]
        
        # perform calculations for the current LSTM cell
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        
        # save the value of the next hidden state, next cell state, and prediction
        a[:,:,t] = a_next
        c[:,:,t]  = c_next
        y[:,:,t] = yt
        caches.append(cache)
    
    # cache stores values needed for backward propagation
    caches = (caches, x)

    return a, y, c, caches



# References
# [1] https://www.coursera.org/learn/nlp-sequence-models/programming/yIJFK/building-your-recurrent-neural-network-step-by-step
