from scipy.io import loadmat
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np
X_raw = loadmat("R2198_20ms.mat")
y_raw = np.loadtxt("R2198_locations.dat")
X_raw = X_raw['mm'].T

def generate_datasets(n, fft):
    print(n, 'fft:', fft)
    X = np.zeros((X_raw.shape[0],2*n*33))
    y = np.zeros(y_raw.shape)
    for i in range(n, X_raw.shape[0]-n):
        if fft:
            X[i] = np.real(np.fft.fft(X_raw[i-n:i+n].T)).flatten()
        else:
            X[i] = X_raw[i-n:i+n].flatten(order='F')
        y[i] = y_raw[i]

    X = X[n:-n]
    y = y[n:-n]

    return train_test_split(X, y, test_size=0.2, random_state=0)

results = []
clf = RandomForestRegressor(n_jobs=-1, verbose=0)

#use a full grid over all parameters
param_grid = {"n_estimators": [10, 50, 100],
              "max_depth": [3, 5, 10, None],
              "max_features": np.arange(1, 11),
              # "min_samples_split": np.arange(2, 11),
              # "min_samples_leaf": np.arange(1, 11),
              "bootstrap": [True, False]}



for time_window in [10,20,50,75]:
    use_fft = True
    X_train, X_test, y_train, y_test = generate_datasets(time_window, fft=use_fft)

    grid_search = GridSearchCV(clf, param_grid=param_grid)
    grid_search.fit(X_train, y_train)
    print grid_search.best_params_, time_window
    results.append({'time_window': time_window,
                    'fft': use_fft,
                    'best_parameters': grid_search.best_params_,
                    'MAE': mean_absolute_error(grid_search.predict(X_test), y_test),
                    'model': clf.__class__.__name__})
print results
