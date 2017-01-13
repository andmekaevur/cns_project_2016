import inspect
import keras
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.callbacks import EarlyStopping

import numpy as np

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

experiment_index = increment_experiment_index()

X_raw = loadmat("R2198_20ms.mat")
y_raw = np.loadtxt("R2198_locations.dat")

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


X_prev, y_prev = generate_datasets()

# X_prev.shape, y_prev.shape

# input shape (nb_samples, timesteps, input_dim)
# output shape (nb_samples, output_dim)
timesteps = 10
X = np.zeros((X_prev.shape[0]-timesteps, timesteps, X_prev.shape[1]))
y = np.zeros((y_prev.shape[0]-timesteps, y_prev.shape[1]))

for i in range(X.shape[0]):
    X[i] = X_prev[i:i+timesteps]
    y[i] = y_prev[i+timesteps]

print('X', X.shape, 'y', y.shape)

test_set_size = 0.2
test_chunk_location = 0.0
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

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

def build_model():
    model = Sequential()
    # keras.layers.recurrent.Recurrent( input_dim=None, input_length=None)

    model.add(LSTM(input_dim=X.shape[2], input_length=timesteps, output_dim=128, return_sequences=False))
    model.add(Activation('sigmoid'))
    model.add(Dense(64))
    model.add(Dense(2))
    # model.add(Dense(2))
    # model.add(Activation('sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='mae', optimizer='sgd')
    return model


model = build_model()
log_model_src()

print('building model')
print('model built')


print('fitting')
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0)
model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stopping])
