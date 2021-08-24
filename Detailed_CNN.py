# The following code was adapted from Week 1 Programming Assignments 1 and 2 in the Convolutional Neural Networks course by DeepLearning.AI offered on Coursera
# https://www.coursera.org/learn/convolutional-neural-networks/home/week/1



import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')



"""
This function implements padding
"""
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values = (0,0))
    return X_pad



np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 3)
print ("x.shape =\n", x.shape)
print ("x_pad.shape =\n", x_pad.shape)
print ("x[1,1] =\n", x[1, 1])
print ("x_pad[1,1] =\n", x_pad[1, 1])



"""
This function implements one calculation of a filter and a slice
"""

def conv_single_step(a_slice_prev, W, b):
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + float(b)
    return Z



"""
This function implements forward propagation for a convolution layer
"""
def conv_forward(A_prev, W, b, hparameters):
    
    # A_prev is output activations from previous layer and input for this layer
    # W and b are weight and bias parameters
    
    # retrieve dimension values from input and weight values
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
    (f, f, n_C_prev, n_C) = np.shape(W)
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    # calculate output dimensions and initialize the output with zeros
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # helper function that pads the input
    A_prev_pad = zero_pad(A_prev, pad)
    
    # loop over the batch of training examples
    for i in range(m):
        a_prev_pad = A_prev[i]
        
        # loop over vertical axis of output
        for h in range(n_H):
            vert_start = stride * h
            vert_end = (stride * h) + f
            
            # loop over horizontal axis of output
            for w in range(n_W):
                horiz_start = stride * w
                horiz_end = (stride * w) + f
                
                # loop over number of filters for output
                for c in range(n_C):
                    
                    # define the section(slice) of the input to perform convolution operation
                    a_slice_prev = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    # select correct weight and bias parameters
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    
                    # helper function that performs the filter and slice calculation
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
                    
                    # apply activation function
                    A[i, h, w, c] = activation(Z[i, h, w, c])
    
    # cache stores information for backward propagation
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache



"""
This function implements forward propagation for a pooling layer
"""
def pool_forward(A_prev, hparameters, mode = "max"):
    
    # A_prev in the input data
    # mode is a string containing the type of pooling
    
    # retrieve dimension values from input
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # calculate output dimensions 
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # initialize output with zeros
    A = np.zeros((m, n_H, n_W, n_C))              

    # loop over the batch of training examples
    for i in range(m):
        
        # loop over vertical axis of output
        for h in range(n_H):
            vert_start = stride * h
            vert_end = (stride * h) + f
            
            # loop over horizontal axis of output
            for w in range(n_W):
                horiz_start = stride * w
                horiz_end = (stride * w) + f
                
                # loop over number of filters for output
                for c in range(n_C):
                    
                    # define section of the input to perform pooling operation
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    # perform max pooling
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                        
                    # perform average pooling
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    # cache stores information for backward propagation
    cache = (A_prev, hparameters)
    
    return A, cache



"""
This function implements backward porpagation for a convolution function
"""
def conv_backward(dZ, cache):
    
    (A_prev, W, b, hparameters) = cache
    
    # retrieve dimensions
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
    (f, f, n_C_prev, n_C) = np.shape(W)
    stride = hparameters['stride']
    pad = hparameters['pad']
    (m, n_H, n_W, n_C) = np.shape(dZ)
    
    # initialize shape of outputs
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C)) 
    
    # perform padding
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    # loop over the training examples
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = stride * h
                    vert_end = (stride * h) + f
                    horiz_start = stride * w
                    horiz_end = (stride * w) + f
                    
                    # define section (slice) from input
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    # calculate gradients
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    
    # ensure output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db



"""
This function creates a mask
"""
def create_mask_from_window(x):
    mask = (x == np.max(x))
    return mask



"""
This function distributes a value through a matrix (used for average pooling)
"""
def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = dz / (n_H * n_W)
    a = np.ones(shape) * average
    return a



"""
This function implements backward propagation for the pooling layer
"""
def pool_backward(dA, cache, mode = "max"):

    (A_prev, hparameters) = cache
    stride = hparameters['stride']
    f = hparameters['f']
    
    # retrieve dimensions
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
    (m, n_H, n_W, n_C) = np.shape(dA)
    dA_prev = np.zeros(np.shape(A_prev))
    
    # loop over the training examples
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    
                    # define corners of the section(slice) of input
                    vert_start = stride * h
                    vert_end = (stride * h) + f
                    horiz_start = stride * w
                    horiz_end = (stride * w) + f
                    if mode == "max":
                        
                        # define slice
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        
                        # create mask to keep track of max
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
                        
                    elif mode == "average":
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        
                        # add distributed value
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev



# References 
# [1] https://www.coursera.org/learn/convolutional-neural-networks/programming/4xt9A/convolutional-model-step-by-step
