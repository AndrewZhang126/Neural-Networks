#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[5]:


"""
This function implements padding
"""
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values = (0,0))
    return X_pad


# In[6]:


np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 3)
print ("x.shape =\n", x.shape)
print ("x_pad.shape =\n", x_pad.shape)
print ("x[1,1] =\n", x[1, 1])
print ("x_pad[1,1] =\n", x_pad[1, 1])

assert type(x_pad) == np.ndarray, "Output must be a np array"
assert x_pad.shape == (4, 9, 9, 2), f"Wrong shape: {x_pad.shape} != (4, 9, 9, 2)"
print(x_pad[0, 0:2,:, 0])
assert np.allclose(x_pad[0, 0:2,:, 0], [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], 1e-15), "Rows are not padded with zeros"
assert np.allclose(x_pad[0, :, 7:9, 1].transpose(), [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], 1e-15), "Columns are not padded with zeros"
assert np.allclose(x_pad[:, 3:6, 3:6, :], x, 1e-15), "Internal values are different"

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0, :, :, 0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0, :, :, 0])
zero_pad_test(zero_pad)


# In[7]:


"""
This function implements one calculation of a filter and a slice
"""

def conv_single_step(a_slice_prev, W, b):
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + float(b)
    return Z


# In[8]:


np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)
conv_single_step_test(conv_single_step)

assert (type(Z) == np.float64 or type(Z) == np.float32), "You must cast the output to float"
assert np.isclose(Z, -6.999089450680221), "Wrong value"


# In[31]:


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


# In[32]:


np.random.seed(1)
A_prev = np.random.randn(2, 5, 7, 4)
W = np.random.randn(3, 3, 4, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad" : 1,
               "stride": 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
print("Z's mean =\n", np.mean(Z))
print("Z[0,2,1] =\n", Z[0, 2, 1])
print("cache_conv[0][1][2][3] =\n", cache_conv[0][1][2][3])

conv_forward_test(conv_forward)


# In[27]:


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


# In[33]:


# Case 1: stride of 1
np.random.seed(1)
A_prev = np.random.randn(2, 5, 5, 3)
hparameters = {"stride" : 1, "f": 3}

A, cache = pool_forward(A_prev, hparameters, mode = "max")
print("mode = max")
print("A.shape = " + str(A.shape))
print("A[1, 1] =\n", A[1, 1])
print()
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A.shape = " + str(A.shape))
print("A[1, 1] =\n", A[1, 1])

pool_forward_test(pool_forward)


# In[34]:


# Case 2: stride of 2
np.random.seed(1)
A_prev = np.random.randn(2, 5, 5, 3)
hparameters = {"stride" : 2, "f": 3}

A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A.shape = " + str(A.shape))
print("A[0] =\n", A[0])
print()

A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A.shape = " + str(A.shape))
print("A[1] =\n", A[1])


# In[50]:


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


# In[51]:


# We'll run conv_forward to initialize the 'Z' and 'cache_conv",
# which we'll use to test the conv_backward function
np.random.seed(1)
A_prev = np.random.randn(10, 4, 4, 3)
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad" : 2,
               "stride": 2}
Z, cache_conv = conv_forward(A_prev, W, b, hparameters)

# Test conv_backward
dA, dW, db = conv_backward(Z, cache_conv)

print("dA_mean =", np.mean(dA))
print("dW_mean =", np.mean(dW))
print("db_mean =", np.mean(db))

assert type(dA) == np.ndarray, "Output must be a np.ndarray"
assert type(dW) == np.ndarray, "Output must be a np.ndarray"
assert type(db) == np.ndarray, "Output must be a np.ndarray"
assert dA.shape == (10, 4, 4, 3), f"Wrong shape for dA  {dA.shape} != (10, 4, 4, 3)"
assert dW.shape == (2, 2, 3, 8), f"Wrong shape for dW {dW.shape} != (2, 2, 3, 8)"
assert db.shape == (1, 1, 1, 8), f"Wrong shape for db {db.shape} != (1, 1, 1, 8)"
assert np.isclose(np.mean(dA), 1.4524377), "Wrong values for dA"
assert np.isclose(np.mean(dW), 1.7269914), "Wrong values for dW"
assert np.isclose(np.mean(db), 7.8392325), "Wrong values for db"

print("\033[92m All tests passed.")


# In[52]:


"""
This function creates a mask
"""

def create_mask_from_window(x):
    mask = (x == np.max(x))
    return mask


# In[53]:


np.random.seed(1)
x = np.random.randn(2, 3)
mask = create_mask_from_window(x)
print('x = ', x)
print("mask = ", mask)

x = np.array([[-1, 2, 3],
              [2, -3, 2],
              [1, 5, -2]])

y = np.array([[False, False, False],
     [False, False, False],
     [False, True, False]])
mask = create_mask_from_window(x)

assert type(mask) == np.ndarray, "Output must be a np.ndarray"
assert mask.shape == x.shape, "Input and output shapes must match"
assert np.allclose(mask, y), "Wrong output. The True value must be at position (2, 1)"

print("\033[92m All tests passed.")


# In[54]:


"""
This function distributes a value through a matrix (used for average pooling)
"""

def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = dz / (n_H * n_W)
    a = np.ones(shape) * average
    return a


# In[55]:


a = distribute_value(2, (2, 2))
print('distributed value =', a)


assert type(a) == np.ndarray, "Output must be a np.ndarray"
assert a.shape == (2, 2), f"Wrong shape {a.shape} != (2, 2)"
assert np.sum(a) == 2, "Values must sum to 2"

a = distribute_value(100, (10, 10))
assert type(a) == np.ndarray, "Output must be a np.ndarray"
assert a.shape == (10, 10), f"Wrong shape {a.shape} != (10, 10)"
assert np.sum(a) == 100, "Values must sum to 100"

print("\033[92m All tests passed.")


# <a name='5-2-3'></a>
# ### 5.2.3 Putting it Together: Pooling Backward 
# 
# You now have everything you need to compute backward propagation on a pooling layer.
# 
# <a name='ex-8'></a>
# ### Exercise 8 - pool_backward
# 
# Implement the `pool_backward` function in both modes (`"max"` and `"average"`). You will once again use 4 for-loops (iterating over training examples, height, width, and channels). You should use an `if/elif` statement to see if the mode is equal to `'max'` or `'average'`. If it is equal to 'average' you should use the `distribute_value()` function you implemented above to create a matrix of the same shape as `a_slice`. Otherwise, the mode is equal to '`max`', and you will create a mask with `create_mask_from_window()` and multiply it by the corresponding value of dA.

# In[62]:


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


# In[63]:


np.random.seed(1)
A_prev = np.random.randn(5, 5, 3, 2)
hparameters = {"stride" : 1, "f": 2}
A, cache = pool_forward(A_prev, hparameters)
print(A.shape)
print(cache[0].shape)
dA = np.random.randn(5, 4, 2, 2)

dA_prev1 = pool_backward(dA, cache, mode = "max")
print("mode = max")
print('mean of dA = ', np.mean(dA))
print('dA_prev1[1,1] = ', dA_prev1[1, 1])  
print()
dA_prev2 = pool_backward(dA, cache, mode = "average")
print("mode = average")
print('mean of dA = ', np.mean(dA))
print('dA_prev2[1,1] = ', dA_prev2[1, 1]) 

assert type(dA_prev1) == np.ndarray, "Wrong type"
assert dA_prev1.shape == (5, 5, 3, 2), f"Wrong shape {dA_prev1.shape} != (5, 5, 3, 2)"
assert np.allclose(dA_prev1[1, 1], [[0, 0], 
                                    [ 5.05844394, -1.68282702],
                                    [ 0, 0]]), "Wrong values for mode max"
assert np.allclose(dA_prev2[1, 1], [[0.08485462,  0.2787552], 
                                    [1.26461098, -0.25749373], 
                                    [1.17975636, -0.53624893]]), "Wrong values for mode average"
print("\033[92m All tests passed.")


# # References 
# [1] https://www.coursera.org/learn/convolutional-neural-networks/programming/4xt9A/convolutional-model-step-by-step
