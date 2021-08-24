#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython

get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# the number of time steps input to the model from the spectrogram
Tx = 5511 

# the number of frequencies input to the model at each time step of the spectrogram
n_freq = 101 


# In[9]:


# the number of time steps in the output of our model
Ty = 1375 


# In[ ]:


# load audio segments using pydub 
activates, negatives, backgrounds = load_raw_audio('./raw_data/')


# In[11]:


"""
This function gets a random time segment
"""
def get_random_time_segment(segment_ms):
    
    segment_start = np.random.randint(low=0, high=10000-segment_ms)   
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)


# In[14]:


"""
This function checks if a segment overlaps with exisitng segments
"""
def is_overlapping(segment_time, previous_segments):
    
    segment_start, segment_end = segment_time
    overlap = False
    
    for previous_start, previous_end in previous_segments: 
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
            break

    return overlap


# In[17]:


"""
This function inserts an audio segment into background noise at a random time step with no overlaps
"""

def insert_audio_clip(background, audio_clip, previous_segments):
   
    segment_ms = len(audio_clip)
    
    # helper function gets random time segment
    segment_time = get_random_time_segment(segment_ms)
    
    # keep picking random time segments until no overlaps occur (maximum 5 times)
    retry = 5 
    while is_overlapping(segment_time, previous_segments) and retry >= 0:
        segment_time = get_random_time_segment(segment_ms)
        retry = retry - 1
   
    # insert segment into the background
    if not is_overlapping(segment_time, previous_segments):
        
        # addd new segment_time to the list of previous segments
        previous_segments.append(segment_time)

        # superpose audio segment and background
        new_background = background.overlay(audio_clip, position = segment_time[0])
    else:
        new_background = background
        segment_time = (10000, 10000)
    
    return new_background, segment_time


# In[32]:


"""
This function sets the 50 time step labels after a segment as ones
"""

def insert_ones(y, segment_end_ms):
   
    _, Ty = y.shape
    
    # duration of the background
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    if segment_end_y < Ty:
        
        # add 1 to the correct indices in the background label
        for i in range(segment_end_y + 1, segment_end_y + 51):
            if i < Ty:
                y[0, i] = 1
    
    return y


# In[45]:


"""
This function creates a single training example
"""
def create_training_example(background, activates, negatives, Ty):
    
    # background is a 10 second background audio recording
    # activates is a list of trigger word audio clips
    # negatives is a list of random audio clips of words that are not the trigger word
    # Ty is the number of time steps in the output
    
    # make background quieter
    background = background - 20

    # initialize the label of each time step as zero 
    y = np.zeros((1, Ty))

    previous_segments = []
    
    # select 0 to 4 random trigger word audio clips
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    
    # loop over randomly selected trigger word clips and insert in background
    for random_activate in random_activates: 
        
        # helper function inserts trigger word clip into background with no overlapping clips
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        
        # retrieve start and end of trigger word segment
        segment_start, segment_end = segment_time
        
        # helper function inserts a label of 1 in 50 time steps after trigger word segment
        y = insert_ones(y, segment_end)

    # select 0-2 random non-trigger word audio clips
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    # loop over randomly selected non-trigger word clips and insert in background
    for random_negative in random_negatives: 
        
        # helper function inserts non-trigger word clip into bakcground with no overlapping clips
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    
    # standardize the volume of the audio clip 
    background = match_target_amplitude(background, -20.0)

    # export new training example 
    file_handle = background.export("train" + ".wav", format="wav")
    
    # plot spectrogram of the new recording
    x = graph_spectrogram("train.wav")
    
    return x, y


# In[ ]:


x, y = create_training_example(backgrounds[0], activates, negatives, Ty)

#plot the labels
plt.plot(y[0])


# In[ ]:


# gemerate small training set
np.random.seed(4543)
nsamples = 32
X = []
Y = []
for i in range(0, nsamples):
    if i%10 == 0:
        print(i)
    x, y = create_training_example(backgrounds[i % 2], activates, negatives, Ty)
    X.append(x.swapaxes(0,1))
    Y.append(y.swapaxes(0,1))
X = np.array(X)
Y = np.array(Y)


# In[53]:


# load preprocessed dev set examples
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")


# In[54]:


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam


# In[55]:


"""
This function creates the model [CONV] -> [GRU] -> [GRU] -> [FULLY CONNECTED]
"""
def modelf(input_shape):
    
    X_input = Input(shape = input_shape)
    
    
    # CONV layer
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
    # batch normalization
    X = BatchNormalization()(X)
    # ReLU activation
    X = Activation("relu")(X)
    # dropout
    X = Dropout(rate=0.8)(X)                           

    # first GRU layer 
    X = GRU(units=128, return_sequences=True)(X)
    # dropout
    X = Dropout(rate=0.8)(X)    
    # batch normalization.
    X = BatchNormalization()(X)                           
    
    # second GRU layer 
    X = GRU(units=128, return_sequences=True)(X)
    # dropout
    X = Dropout(rate=0.8)(X)           
    # batch normalization
    X = BatchNormalization()(X)    
    # dropout
    X = Dropout(rate=0.8)(X)                                  
    
    # time-distributed fully connected layer 
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) 

    model = Model(inputs = X_input, outputs = X)
    
    return model  


# In[57]:


model = modelf(input_shape = (Tx, n_freq))


# In[59]:


from tensorflow.keras.models import model_from_json

# load pre-trained model
json_file = open('./models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('./models/model.h5')


# In[61]:


opt = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])


# In[ ]:


model.fit(X, Y, batch_size = 16, epochs=1)


# In[ ]:


# test model
loss, acc, = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)


# In[64]:


# run audio through the network
def detect_triggerword(filename):
    plt.subplot(2, 1, 1)
    audio_clip = AudioSegment.from_wav(filename)
    audio_clip = match_target_amplitude(audio_clip, -20.0)
    file_handle = audio_clip.export("tmp.wav", format="wav")
    filename = "tmp.wav"

    x = graph_spectrogram(filename)
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions


# In[65]:


chime_file = "audio_examples/chime.wav"
"""
This funvtion implements a "chiming" sound to play when the model detects the trigger word
"""
def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    
    # initialize the number of consecutive output steps
    consecutive_timesteps = 0
    
    # loop over the output steps 
    for i in range(Ty):
        
        # increment consecutive output steps
        consecutive_timesteps += 1
        
        # check if prediction is higher than the threshold and more than 20 consecutive output steps have passed
        if consecutive_timesteps > 20:
            
            # superpose audio and background
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds) * 1000)
            
            # reset consecutive output steps to 0
            consecutive_timesteps = 0
            
        # if amplitude is smaller than the threshold reset the counter
        if predictions[0, i, 0] < threshold:
            consecutive_timesteps = 0
        
    audio_clip.export("chime_output.wav", format='wav')


# In[ ]:


# test model on audio clips
filename = "./raw_data/dev/1.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)
IPython.display.Audio("./chime_output.wav")


# # References
# [1] https://www.coursera.org/learn/nlp-sequence-models/programming/MHAsK/trigger-word-detection
