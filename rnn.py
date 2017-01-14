#!/home/hpc_robertr/virtualenv3/bin/python3

import sys
import pickle

import inspect
import keras
from scipy.io import loadmat

from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.callbacks import EarlyStopping

import numpy as np
np.random.seed(0) 

def log_model_src():
    with open('rnn_experiments/model-source/{}.py'.format(experiment_index), 'w') as f:
        f.write(inspect.getsource(build_model))

def increment_experiment_index():
    with open('rnn_experiments/experiment-index', 'r+') as f:
        new_index = str(int(f.readline()) + 1)
        f.seek(0)
        f.write(new_index)
    print('Experiment #', new_index)
    return new_index

# experiment_index = increment_experiment_index()

args = list(map(int, sys.argv[1:]))

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
# i, window, step, timestep, rnn_data_type, nodes
experiment_index = args[0]
# window = args[1]
# step = args[2]
# timestep = args[3]
# rnn_data_type = args[4]
# nodes = args[5]
# args = list(map(int, sys.argv))

X_raw = loadmat("cns_project_2016/R2198_20ms.mat")
y_raw = np.loadtxt("cns_project_2016/R2198_locations.dat")

X_raw = X_raw['mm'].T


def generate_datasets(window=50, step=25):
    print('generating data')
    print('window {}, step {}'.format(window,step))
    dataset_length = int(X_raw.shape[0]/step)
    X = np.zeros((dataset_length, 2*window*33))
    y = np.zeros((dataset_length, 2))

    for filtered_i, data_i in enumerate(range(window, X_raw.shape[0]-window, step)):
        X[filtered_i] = np.real(np.fft.fft(X_raw[data_i-window:data_i+window].T)).flatten()
        y[filtered_i] = y_raw[data_i]
    
    return X, y


X_prev, y_prev = generate_datasets(window=args[1], step=args[2])

# X_prev.shape, y_prev.shape

# input shape (nb_samples, timesteps, input_dim)
# output shape (nb_samples, output_dim)
timesteps = args[3]
X = np.zeros((X_prev.shape[0]-timesteps, timesteps, X_prev.shape[1]))
y = np.zeros((y_prev.shape[0]-timesteps, y_prev.shape[1]))

rnn_data_type = args[4]
print('rnn_data_type', rnn_data_type)
if rnn_data_type == 1:
    for i in range(X.shape[0]):
        X[i] = X_prev[i:i+timesteps]
        y[i] = y_prev[i+timesteps]
if rnn_data_type == 2:
    for i in range(X.shape[0]):
        X[i] = X_prev[i:i+timesteps]
        y[i] = y_prev[i+timesteps-1]
if rnn_data_type == 3:
    X = np.zeros((X_prev.shape[0]-2*timesteps, 2*timesteps, X_prev.shape[1]))
    y = np.zeros((y_prev.shape[0]-2*timesteps, y_prev.shape[1]))
    for i in range(X.shape[0]):
        idx = np.hstack([np.arange(i, i+timesteps), np.flipud(np.arange(i+timesteps, i+2*timesteps))])
        X[i] = X_prev[idx]
        y[i] = y_prev[i+timesteps]



print('X', X.shape, 'y', y.shape)

# exit
test_set_size = 0.2
test_chunk_location = 0.4
dataset_length = len(X)
print('sampling with a chunk, size {}, location {}'.format(test_set_size, test_chunk_location))
test_set_start = int(dataset_length*test_chunk_location)
test_set_end = test_set_start + int(dataset_length*test_set_size)
print('indexes {}:{}'.format(test_set_start, test_set_end))
test_chunk = np.arange(test_set_start, test_set_end)
X_val = X[test_chunk]
X_train = np.delete(X, test_chunk, axis=0)
y_val = y[test_chunk]
y_train = np.delete(y, test_chunk, axis=0)


# first batch experiments, optimizer: rmsprop
# second batch: sgd

def build_model():
    model = Sequential()
    # keras.layers.recurrent.Recurrent( input_dim=None, input_length=None)

    model.add(LSTM(input_dim=X.shape[2], input_length=X.shape[1], output_dim=args[5], return_sequences=False))
    model.add(Activation('sigmoid'))
    model.add(Dense(args[6]))
    model.add(Dense(2))

    # try using different optimizers and different optimizer configs
    model.compile(loss='mae', optimizer='sgd')
    return model

# nodes = args[6]
model = build_model()
# log_model_src()

print('building model')
print('model built')


print('fitting')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.4, patience=2, verbose=0)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stopping], nb_epoch=20)
print(history.history)
history.history['args'] = args

pickle.dump(history.history, open( "rnn_experiments/{}.p".format(experiment_index), "wb"))
