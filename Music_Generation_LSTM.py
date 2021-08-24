# The following code was adapted from Week 1 Programming Assignment 3 in the Sequence Models course by DeepLearning.AI offered on Coursera
# https://www.coursera.org/learn/nlp-sequence-models/home/week/1



import IPython
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical



#preprocess raw music data into values
X, Y, n_values, indices_values, chords = load_music_utils('data/original_metheny.mid')



# number of dimensions for the hidden state of each LSTM cell.
n_a = 64 

# number of music values
n_values = 90 

# reshape layer instance
reshaper = Reshape((1, n_values))

# LSTM layer instance
LSTM_cell = LSTM(n_a, return_state = True)   

# fully connected layer instance
densor = Dense(n_values, activation='softmax') 



"""
This function implements an RNN model where each cell is [RESHAPE -> LSTM -> FULLY CONNECTED]
"""
def djmodel(Tx, LSTM_cell, densor, reshaper):
   
    n_values = densor.units
    n_a = LSTM_cell.units
    
    # define the shape of the input layer
    X = Input(shape=(Tx, n_values)) 
    
    # define initial hidden and cell state
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    outputs = []
    
    # loop over the entire sequence
    for t in range(Tx):
        
        x = X[:, t, :]
        x = reshaper(x)
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
      
    # create model instance
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    
    return model



# create the model
model = djmodel(Tx=30, LSTM_cell=LSTM_cell, densor=densor, reshaper=reshaper)



# compile the model
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])



# train the model
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
history = model.fit([X, a0, c0], list(Y), epochs=100, verbose = 0)



"""
This function implements a model that generates a sequence of musical values
"""
def music_inference_model(LSTM_cell, densor, Ty=100):

    # LSTM_cell is the trained LSTM layer
    # densor is the trained fully connected layer
    
    n_values = densor.units
    n_a = LSTM_cell.units
    
    # define the shape of the input
    x0 = Input(shape=(1, n_values))
    
    
    # define initial hidden and cell state 
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    outputs = []
    
    # loop over the number of time steps to generate
    for t in range(Ty):
        
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
 
        # convert last output into an input for next time step
        x = tf.math.argmax(out, axis=-1)
        x = tf.one_hot(x, depth=n_values)
        x = RepeatVector(1)(x)
        
    # create model instance
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    
    return inference_model



# define the model
inference_model = music_inference_model(LSTM_cell, densor, Ty = 50)



x_initializer = np.zeros((1, 1, n_values))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))



"""
This function predicts the next values using the inference model
"""
def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    
    n_values = x_initializer.shape[2]
    
    # use inference model to predict an output sequence 
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    
    # convert prediction into an array of maximum probabilities indices
    indices = np.array(np.argmax(pred, axis=2))
    
    # convert indices to one-hot vectors
    results = to_categorical(indices, num_classes=90)
 
    return results, indices



# generate and record music
out_stream = generate_music(inference_model, indices_values, chords)



# # References
# [1] https://www.coursera.org/learn/nlp-sequence-models/programming/ZS7X2/jazz-improvisation-with-lstm
# [2] https://github.com/jisungk/deepjazz
# [3] http://ai.stanford.edu/~kdtang/papers/smc09-jazzgrammar.pdf
# [4] http://smc07.uoa.gr/SMC07%20Proceedings/SMC07%20Paper%2055.pdf
# [5] http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.5.7473&rep=rep1&type=pdf
