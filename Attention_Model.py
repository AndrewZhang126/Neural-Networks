# The following code was adapted from Week 3 Programming Assignment 1 in the Sequence Models course by DeepLearning.AI offered on Coursera
# https://www.coursera.org/learn/nlp-sequence-models/home/week/3



from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



# load dataset
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

Tx = 30
Ty = 10

# preprocess data
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)



# repeat values
repeator = RepeatVector(Tx)

# concatenate values
concatenator = Concatenate(axis=-1)

# fully connected layers with activations
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")

# activation layer
activator = Activation(softmax, name='attention_weights') 

# compute dot product
dotor = Dot(axes = 1)



"""
This function performs one step of the attention mechanism
"""
def one_step_attention(a, s_prev):

    # s-prev is the previous hidden state of the post-attention LSTM cell
    # a is the hidden state output of the pre-attention Bi_LSTM cell
    
    # Use repeator to repeat s_prev for concatenating
    s_prev = repeator(s_prev)
    
    # concatenate a and s_prev
    concat = concatenator([a, s_prev])
    
    # apply fully connected layers
    e = densor1(concat)
    energies = densor2(e)
    
    # apply activation to compute the attention weights 
    alphas = activator(energies)
    
    # compute dot product of attention weights and hidden state to pass onto next post-attention LSTM-cell cell
    context = dotor([alphas, a])
    
    return context



# number of units for "a" and "s" hidden states
n_a = 32 
n_s = 64 



# post-attention LSTM layer
post_activation_LSTM_cell = LSTM(n_s, return_state = True) 

# final fully connected layer with activation
output_layer = Dense(len(machine_vocab), activation=softmax)



"""
This function implements the attention model
"""
def modelf(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
 
    # Tx is the length of the input sequence
    # Ty is the length of the output sequence
    
    # define the shape of the input, initial hidden state s0 and inital cell state c0
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    outputs = []
    
    # define pre-attention Bi-LSTM
    a = Bidirectional(LSTM(units=n_a, return_sequences=True))(X)
    
    # loop through length of output sequence
    for t in range(Ty):
    
        # perform one step of the attention mechanism 
        context = one_step_attention(a, s)
        
        # apply the post-attention LSTM cell to the output of the attention mechanism
        s, _, c = post_activation_LSTM_cell(inputs=context, initial_state=[s, c])
        
        # apply fully connected layer
        out = output_layer(inputs=s)
        
        outputs.append(out)
    
    # create model instance
    model = Model(inputs=[X, s0, c0], outputs=outputs)
    
    return model



# create the model
model = modelf(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))



# compile the model
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])



s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))



model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)



# load pre-trained weights
model.load_weights('models/model.h5')



# print results
EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
s00 = np.zeros((1, n_s))
c00 = np.zeros((1, n_s))
for example in EXAMPLES:
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    source = np.swapaxes(source, 0, 1)
    source = np.expand_dims(source, axis=0)
    prediction = model.predict([source, s00, c00])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    print("source:", example)
    print("output:", ''.join(output),"\n")



# plot attention map
attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64);


# References
# [1] https://www.coursera.org/learn/nlp-sequence-models/programming/L0BBe/neural-machine-translation
