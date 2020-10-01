import numpy as np
import math
import os

from keras import backend as K
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.wrappers import  Bidirectional
from keras.callbacks import LearningRateScheduler
from keras import regularizers
import pickle
import commpy.channelcoding.convcode as cc
from commpy.utilities import *
from gen_channel_data import *

import sys
import argparse

from examples import * # Generating pairs of (noisy codewords, true messages)

import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('use python -W ignore xxx.py to ignore warnings')


################################
# Arguments
################################

def get_args():
   
    parser = argparse.ArgumentParser()

    parser.add_argument('-if_training', type=bool, default=True)
    parser.add_argument('-path_trained_model', type=str, default="TrainedModel_75.h5")

    # NN decoder parameters 
    parser.add_argument('-rx_type', choices=['rnn-lstm', 'rnn-gru', 'cnn'], default='cnn')
    parser.add_argument('-num_rx_layer', type=int, default=4)
    parser.add_argument('-rx_num_filter', type=int, default=64)
    parser.add_argument('-rnn_rx_dir', choices=['sd','bd'], default='bd')
    parser.add_argument('-rnn_n_unit', type=int, default=50)

    args = parser.parse_args()

    return args

args = get_args()

print('args', args)


training = args.if_training # If False: load trained model and just test. If True: train. 


# Network Parameters
test_ratio = 1
num_epoch = 12
learning_rate = 1e-2                # one suggestion is LR 0.001 and batch size 10

train_batch_size = 32              # Many2Many better use small batch (10)
test_batch_size  = 100

# Tx Encoder Parameters
code_rate   =  2                     #  2,3

k = 2000000                            # Number of total message bits for training. 
step_of_history = 10                # Length of each message bit sequence
k_test = 1000000                        # Number of total message bits for testing. 


# Rx Decoder Parameters
rx_direction          = args.rnn_rx_dir         #'bd', 'sd'
num_hunit_rnn_rx      = args.rnn_n_unit         # rnn param

# print parameters

print('*'*100)
print('Message bit is ', k)
print('learning rate is ', learning_rate)
print('batch size is ', train_batch_size)
print('step of history is ', step_of_history)
print('The RNN has ', rx_direction, args.rx_type, 'with ', args.num_rx_layer, 'layers with ', num_hunit_rnn_rx, ' unit')
print('*'*100)

def errors(y_true, y_pred):
    ErrorTensor = K.not_equal(K.round(y_true), K.round(y_pred))
    return K.mean(tf.cast(ErrorTensor, tf.float32))

# Setup LR decay
def scheduler(epoch):

    if epoch > 4 and epoch <=6:
        print('changing by /10 lr')
        lr = learning_rate/10.0
    elif epoch > 6 and epoch <=9:
        print('changing by /100 lr')
        lr = learning_rate/100.0
    elif epoch > 9 and epoch <=12:
        print('changing by /100 lr')
        lr = learning_rate/100.0
    else:
        lr = learning_rate

    return lr
    
change_lr = LearningRateScheduler(scheduler)



##########################################
# Create a NN decoder 
##########################################



inputs = Input(shape=(step_of_history, code_rate)) # For each batch (one noisy codeword), input is of size (message length, 1)
x = inputs # Noisy codeword 

### Defining Decoder Architecture 
if args.rx_type == 'rnn-gru':
    for layer in range(args.num_rx_layer):
        if rx_direction == 'bd':
            x = Bidirectional(GRU(units=num_hunit_rnn_rx, activation='tanh',
                                    return_sequences=True))(x)
            x = BatchNormalization()(x)
        else:
            x = GRU(units=num_hunit_rnn_rx, activation='tanh',
                    return_sequences=True)(x)
            x = BatchNormalization()(x)

elif args.rx_type == 'cnn':
    for layer in range(args.num_rx_layer):
        x = Conv1D(filters=args.rx_num_filter, kernel_size=3, strides=1, activation='relu', padding='same')(x)
        x = BatchNormalization()(x) 
else:
    print('not supported')

predictions = TimeDistributed(Dense(1, activation='sigmoid'))(x) # Soft decoding. For each batch, prediction is of size (message length, 1)

model = Model(inputs=inputs, outputs=predictions) # model denotes a NN that maps inputs to predictions. 


### Optimizer, loss, and evaluation metrics
optimizer= tf.keras.optimizers.Adam(lr=learning_rate,clipnorm=1.)
model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=[errors]) # loss='mean_squared_error' is another option. 

### Summary of NN model 
print(model.summary())


##########################################
# Train a model or load pre-trained model 
##########################################

if training == False: # Load pre-trained model 
        print("Loading trained model from ", args.path_trained_model)
        model.load_weights(args.path_trained_model)
        
else: 
    # Generate training examples
    #noisy_codewords, true_messages, _ = generate_examples(k_test=k, step_of_history=200, SNR=0, code_rate = 2) 

    all_types=  [{'noise_type': 'awgn', 'snr': -5.0}, {'noise_type': 'awgn', 'snr': -3.0}, {'noise_type': 'awgn', 'snr': -1.0}, {'noise_type': 'awgn', 'snr': 1.0}, {'noise_type': 'awgn', 'snr': 3.0}, {'noise_type': 'awgn', 'snr': 5.0}]


    noisy_codewords,true_messages = [], []

    for i in range((2*10**6)):
        index = int(i%len(all_types))
        batch_criteria = all_types[index]
        k_seed = np.random.randint(1, 999999)
        print("iter ", i, k_seed, " index ", index, " b a", batch_criteria )
        n, t = generate_viterbi_batch(batch_size= int(5), \
                                block_len= 10,
                                batch_criteria = batch_criteria,
                                seed = k_seed)
        noisy_codewords.append(n)
        true_messages.append(t)

    noisy_codewords = np.reshape(np.asarray(noisy_codewords), (-1, 10, 2))
    
    true_messages = np.reshape(np.asarray(true_messages), (-1, 10))
    print("shapes", np.shape(noisy_codewords), np.shape(true_messages))
    print('Training examples are generated')

    train_history = model.fit(x=noisy_codewords, y=true_messages, batch_size=train_batch_size,
                callbacks=[change_lr],
                epochs=num_epoch, validation_split=0.1)  # starts training

    model.save_weights('AWGN55.h5')
    


##########################################
# Testing on various SNRs
#########################################


## SNR
SNR_dB_start_Eb = 0
SNR_dB_stop_Eb = 6
SNR_points = 7

TestSNRS = np.linspace(SNR_dB_start_Eb, SNR_dB_stop_Eb, SNR_points, dtype = 'float32')
test_sigmas = 10**(-TestSNRS*1.0/20)

snr_collect = []
ber_collect = []
bler_collect = []

all_types = [{'noise_type': 'awgn', 'snr': -10.0}, {'noise_type': 'awgn', 'snr': -9.0}, {'noise_type': 'awgn', 'snr': -8.0}, {'noise_type': 'awgn', 'snr': -7.0}, {'noise_type': 'awgn', 'snr': -6.0}, {'noise_type': 'awgn', 'snr': -5.0}, {'noise_type': 'awgn', 'snr': -4.0}, {'noise_type': 'awgn', 'snr': -3.0}, {'noise_type': 'awgn', 'snr': -2.0}, {'noise_type': 'awgn', 'snr': -1.0}, {'noise_type': 'awgn', 'snr': 0.0}, {'noise_type': 'awgn', 'snr': 1.0}, {'noise_type': 'awgn', 'snr': 2.0}, {'noise_type': 'awgn', 'snr': 3.0}, {'noise_type': 'awgn', 'snr': 4.0}, {'noise_type': 'awgn', 'snr': 5.0}, {'noise_type': 'awgn', 'snr': 6.0}, {'noise_type': 'awgn', 'snr': 7.0}, {'noise_type': 'awgn', 'snr': 8.0}, {'noise_type': 'awgn', 'snr': 9.0}, {'noise_type': 'awgn', 'snr': 10.0}, {'noise_type': 'memory', 'snr': 6.0, 'alpha': 0.1}, {'noise_type': 'memory', 'snr': 6.0, 'alpha': 0.2}, {'noise_type': 'memory', 'snr': 6.0, 'alpha': 0.3}, {'noise_type': 'memory', 'snr': 6.0, 'alpha': 0.4}, {'noise_type': 'memory', 'snr': 6.0, 'alpha': 0.5}, {'noise_type': 'memory', 'snr': 6.0, 'alpha': 0.6}, {'noise_type': 'memory', 'snr': 6.0, 'alpha': 0.7}, {'noise_type': 'memory', 'snr': 6.0, 'alpha': 0.8}, {'noise_type': 'memory', 'snr': 6.0, 'alpha': 0.9}, {'noise_type': 'multipath', 'snr': 6.0, 'weight': 0}, {'noise_type': 'multipath', 'snr': 6.0, 'weight': 0.05}, {'noise_type': 'multipath', 'snr': 6.0, 'weight': 0.1}, {'noise_type': 'multipath', 'snr': 6.0, 'weight': 0.15}, {'noise_type': 'multipath', 'snr': 6.0, 'weight': 0.2}, {'noise_type': 'multipath', 'snr': 6.0, 'weight': 0.25}, {'noise_type': 'multipath', 'snr': 6.0, 'weight': 0.3}, {'noise_type': 'multipath', 'snr': 6.0, 'weight': 0.35}, {'noise_type': 'multipath', 'snr': 6.0, 'weight': 0.4}, {'noise_type': 'multipath', 'snr': 6.0, 'weight': 0.45}, {'noise_type': 'multipath', 'snr': 6.0, 'weight': 0.5}, {'noise_type': 'bursty', 'snr': 6.0, 'snrb': -22}, {'noise_type': 'bursty', 'snr': 6.0, 'snrb': -20}, {'noise_type': 'bursty', 'snr': 6.0, 'snrb': -18}, {'noise_type': 'bursty', 'snr': 6.0, 'snrb': -16}, {'noise_type': 'bursty', 'snr': 6.0, 'snrb': -14}, {'noise_type': 'bursty', 'snr': 6.0, 'snrb': -12}, {'noise_type': 'bursty', 'snr': 6.0, 'snrb': -10}, {'noise_type': 'bursty', 'snr': 6.0, 'snrb': -8}, {'noise_type': 'bursty', 'snr': 6.0, 'snrb': -6}, {'noise_type': 'bursty', 'snr': 6.0, 'snrb': -4}, {'noise_type': 'bursty', 'snr': 6.0, 'snrb': -2}]

for idx in range(len(all_types)):
    #TestSNR = TestSNRS[idx] 

    #noisy_codewords, true_messages, target = generate_examples(k_test=k_test,step_of_history=step_of_history,SNR=TestSNR) # target: true messages reshaped 

    noisy_codewords,true_messages = [], []
    batch_criteria = all_types[index]
    for i in range(1000):
        k_seed = np.random.randint(1, 999999)
        #print("TEST iter ", i, k_seed, " index ", index, " b a", batch_criteria )
        n, t = generate_viterbi_batch(batch_size= int(5), \
                                block_len= 10,
                                batch_criteria = batch_criteria,
                                seed = k_seed)
        noisy_codewords.append(n)
        true_messages.append(t)

    noisy_codewords = np.reshape(np.asarray(noisy_codewords), (-1, 10, 2))

    true_messages = np.reshape(np.asarray(true_messages), (-1, 10))
    #print("shapes", np.shape(noisy_codewords), np.shape(true_messages))
    #print("test samples gen done")

    preds = np.round(model.predict(noisy_codewords, batch_size=test_batch_size))
   
    b, h, w = np.shape(preds)
    preds = preds.reshape(b,h)
    print(np.shape(preds), " total n", (preds.shape[0] * preds.shape[1]) ) 
    ber = 1- sum(sum(preds == true_messages))*\
           1.0/(preds.shape[0] * preds.shape[1])

    target = true_messages
    tp0 = (abs(np.round(preds)-target)).reshape([target.shape[0],target.shape[1]])    
    bler = sum(np.sum(tp0,axis=1)>0)*1.0/(target.shape[0])# model.evaluate(X_feed_test, X_message_test, batch_size=10)

    print('*** SNR', batch_criteria)
    print('test nn ber', ber)
    print('test nn bler', bler)    

    snr_collect.append(batch_criteria)
    ber_collect.append(ber)
    bler_collect.append(bler)


print('snr_collect')
print(snr_collect)

print('ber_collect')
print(ber_collect)

print('bler_collect')
print(bler_collect)

# Viterbi performance
np.save("ber_awgn55", ber_collect)
viterbi_snr_collect = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
viterbi_ber_collect = [0.0929, 0.0431, 0.0151, 0.0040, 0.0008, 0.0002, 2.92e-05]
viterbi_bler_collect = [0.9961, 0.9341, 0.6685, 0.2979, 0.0872, 0.0198, 0.0044]

"""
plt.plot(snr_collect,ber_collect,'r', label='NN')
plt.plot(viterbi_snr_collect,viterbi_ber_collect,'b', label='Vitebi')
plt.xlabel('SNR')
plt.ylabel('BER')
plt.legend(loc='upper right')
plt.yscale('log')
plt.show()


plt.plot(snr_collect,bler_collect,'r', label='NN')
plt.plot(viterbi_snr_collect,viterbi_bler_collect,'b', label='Viterbi')
plt.xlabel('SNR')
plt.ylabel('BLER')
plt.legend(loc='upper right')
plt.yscale('log')
plt.show()
"""
