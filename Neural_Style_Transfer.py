#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import pprint
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


pp = pprint.PrettyPrinter(indent=4)
img_size = 400

# import pre-trained VGG-19 model from https://github.com/fchollet/deep-learning-models/releases
vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

vgg.trainable = False
pp.pprint(vgg)


# In[ ]:


"""
This function implements the content cost function
"""
def compute_content_cost(content_output, generated_output):
   
    a_C = content_output[-1]
    a_G = generated_output[-1]
    
    # retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # reshape a_C and a_G for cost calculation
    a_C_unrolled = tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, n_H * n_W, n_C])
    
    # compute the cost 
    J_content =  (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    
    return J_content


# In[7]:


"""
This function computes the Gram matrix
"""
def gram_matrix(A):
    GA = tf.linalg.matmul(A, tf.transpose(A))
    return GA


# In[ ]:


# define the layers to use in the style cost function
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]


# In[9]:


"""
This function implements the style cost function for a single layer
"""
def compute_layer_style_cost(a_S, a_G):
    
    # retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # eeshape the images for cost calculation
    a_S = tf.transpose(tf.reshape(a_S, shape=[n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[n_H * n_W, n_C]))

    # helper function that computes the Gram matrix
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    
    # compute the cost
    J_style_layer = (1 / (4 * pow(n_H, 2) * pow(n_W * n_C, 2))) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    
    return J_style_layer



"""
This function computes the overall style cost function from certain layers
"""
def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    
    # initialize the overall style cost
    J_style = 0

    # the last element of the array contains the content layer image, which is not used.
    a_S = style_image_output[:-1]
    a_G = generated_image_output[:-1]
    
    # compute the overall style cost from all selected layers
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # incorporate the weight into style cost
        J_style += weight[1] * J_style_layer

    return J_style


# In[15]:


"""
This function implements the total cost function
"""
@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
   
    J = (alpha * J_content) + (beta * J_style)
    
    return J


# In[ ]:


# load content image
content_image = np.array(Image.open("images/iamge_name.jpg").resize((img_size, img_size)))
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))


# In[ ]:


style_image =  np.array(Image.open("images/image_name.jpg").resize((img_size, img_size)))
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))


# In[ ]:


# randomly initialze image to be generated (slightly correlated to content image)
generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise = tf.random.uniform(tf.shape(generated_image), 0, 0.5)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)


# In[ ]:


# load pre-trained VGG-19 model
def get_layer_outputs(vgg, layer_names):
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# specify content and style layers 
# typically middle layers work best for the content
content_layer = [('block5_conv4', 1)]
vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

# stores output from content and style layers
content_target = vgg_model_outputs(content_image) 
style_targets = vgg_model_outputs(style_image)  

# assign content image to be the input of the model
preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)

a_G = vgg_model_outputs(generated_image)

# compute the content cost
J_content = compute_content_cost(a_C, a_G)

# assign style image to be the input of the model
preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)

# compute the style cost
J_style = compute_style_cost(a_S, a_G)


# In[25]:


"""
This function truncates pixels to be between 0 and 1
"""
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)



"""
This function converts a tensor into an image
"""
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


# In[26]:


"""
This function implements a training step for transfer learning
"""
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:
        
        a_G = vgg_model_outputs(generated_image)
        J_style = compute_style_cost(a_S, a_G)
        J_content = compute_content_cost(a_C, a_G)
        
        # compute the total cost
        J = total_cost(J_content, J_style)

    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))

    return J


# In[ ]:


# train the model
epochs = 2501
for i in range(epochs):
    train_step(generated_image)
    if i % 250 == 0:
        print(f"Epoch {i} ")
    if i % 250 == 0:
        image = tensor_to_image(generated_image)
        imshow(image)
        image.save(f"output/image_{i}.jpg")
        plt.show() 


# In[ ]:


# show the content, stlye, and generated images
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
imshow(content_image[0])
ax.title.set_text('Content image')
ax = fig.add_subplot(1, 3, 2)
imshow(style_image[0])
ax.title.set_text('Style image')
ax = fig.add_subplot(1, 3, 3)
imshow(generated_image[0])
ax.title.set_text('Generated image')
plt.show()


# # References
# [1] https://www.coursera.org/learn/convolutional-neural-networks/programming/4AZ8P/art-generation-with-neural-style-transfer
# [2] https://arxiv.org/abs/1508.06576
# [3] https://harishnarayanan.org/writing/artistic-style-transfer/
# [4] http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
# [5] https://arxiv.org/pdf/1409.1556.pdf
# [6] https://www.vlfeat.org/matconvnet/pretrained/
