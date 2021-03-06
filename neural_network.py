
from scipy.io import loadmat
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
X_raw = loadmat("R2198_20ms.mat")
y_raw = np.loadtxt("R2198_locations.dat")
X_raw = X_raw['mm'].T

def generate_datasets(window, step=None, test_set_sampling='chunk', test_set_size=0.2, test_chunk_location=0.8):
    print('generating data')
    print('window {}, step {}'.format(window,step))
    # step = int(round(window/2))
    step = window
    dataset_length = int(X_raw.shape[0]/step)
    X = np.zeros((dataset_length, 2*window*33))
    y = np.zeros((dataset_length, 2))

    for filtered_i, data_i in enumerate(range(window, X_raw.shape[0]-window, step)):
        X[filtered_i] = np.real(np.fft.fft(X_raw[data_i-window:data_i+window].T)).flatten()
        y[filtered_i] = y_raw[data_i]

    if test_set_sampling == 'uniform_chunks':
        print('uniform chunks')
        test_set_size = 0.1
        chunk_size = int(2*window/step)
        test_set_length = int(dataset_length*test_set_size)
        chunks_count = int(test_set_length/chunk_size)
        chunk_step = int((dataset_length-chunk_size)/chunks_count)
        test_chunks = [list(range(i*chunk_step, i*chunk_step + chunk_size)) for i in range(chunks_count)]
        discard_chunks = [list(range(i*chunk_step - chunk_size, i*chunk_step)) for i in range(chunks_count)][1:]
        test_chunks = sum(test_chunks, [])
        discard_chunks = sum(discard_chunks, [])
        test_and_discard_chunks = test_chunks + discard_chunks
        X_test = X[test_chunks]
        X_train = np.delete(X, test_and_discard_chunks, axis=0)
        y_test = y[test_chunks]
        y_train = np.delete(y, test_and_discard_chunks, axis=0)

    if test_set_sampling == 'chunk':
        print('sampling with a chunk, size {}, location {}'.format(test_set_size, test_chunk_location))
        test_set_start = int(dataset_length*test_chunk_location)
        test_set_end = test_set_start + int(dataset_length*test_set_size)
        print('indexes {}:{}'.format(test_set_start, test_set_end))
        X_test = X[test_set_start:test_set_end]
        X_train = np.delete(X, np.arange(test_set_start, test_set_end), axis=0)
        y_test = y[test_set_start:test_set_end]
        y_train = np.delete(y, np.arange(test_set_start, test_set_end), axis=0)
    if test_set_sampling == 'random':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=0)

    return X_train, X_test, y_train, y_test


results = []

clf = MLPRegressor(max_iter = 500, verbose=0, random_state=1)

#use a full grid over all parameters
param_grid = {"alpha": [0.000001, 0.001,  0.1, 10, 1000],
              "hidden_layer_sizes": [(100,), (300,), (500,)] ,
              "solver": ["sgd", "adam"],
              "max_iter":[200, 1000],
              "activation": ['tanh','logistic'],
              "warm_start": [True, False]}


scaler = StandardScaler()
for time_window in [10, 20, 50,75]:
    use_fft = True
    X_train, X_test, y_train, y_test = generate_datasets(time_window)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # clf.fit(X_train, y_train)
    # mae = mean_absolute_error(clf.predict(X_test), y_test)
    # print mae
    grid_search =  GridSearchCV(clf, param_grid=param_grid)
    grid_search.fit(X_train, y_train)
    print grid_search.best_params_, time_window
    mae = mean_absolute_error(grid_search.predict(X_test), y_test)
    print mae
    results.append({'time_window': time_window,
                    'fft': use_fft,
                    'best_parameters': grid_search.best_params_,
                    'MAE': mae,
                    'model': clf.__class__.__name__})
print results


