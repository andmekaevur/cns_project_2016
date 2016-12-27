## PREPARE DATA FOR CONDUCTING AN EXPERIMENT AND STORE IT ##  

rm(list = ls())
library("R.matlab")
library(randomForest)

# Reading given data
# X_raw:
# - rows are EEG signals for different channels 
# - columns are measurements for different experiments with 20 ms interval
X_raw = readMat("./../R2198_20ms.mat")[["mm"]]
# Y_raw: 
# - columns are coordinates of rat"s position (x, y) 
Y_raw = read.table("./../R2198_locations.dat")

# prepare data sets for condacting an experiment
# feature vector for experiment t is glued togeather timeslices [t - n; t + n] of different channels
# IN: n (int) - size of timewindow; fft (bool) - whether to apply fft or not before concatenation of channel"s timeslices
# OUT: list where
# X - matrix of feature vectors (row is one feature vector); Y - correspondent coordinates
prepare_datasets = function (n, fft) {
  experiments_number = ncol(X_raw)
  channel_number = nrow(X_raw)
  X = matrix(nrow = experiments_number - 2 * n, ncol = (2 * n + 1) * channel_number) 
  Y = matrix(nrow = experiments_number - 2 * n, ncol = 2)
  for (i in (n + 1) : (experiments_number - n)) {
    sample = c()
    for (j in 1 : channel_number) {
      if (fft) {
        window = fft(X_raw[j, (i - n) : (i + n)])
      } else {
        window = X_raw[j, (i - n) : (i + n)]  
      }
      sample = c(sample, window)
    }
    # as i starts from the point which takes into an account n shift
    X[i - n, ] = sample
    Y[i - n, ] = as.matrix(Y_raw[i, ])
  }
  
  return(list(X = X, Y = Y))
}

# Save prepared datasets 
timewindow = c(10, 20, 50, 75)
folder_path = "Data/"
for (n in timewindow) {
  res_raw = prepare_datasets(n, FALSE)
  res_fft = prepare_datasets(n, TRUE)
  saveRDS(res_raw, paste(folder_path, "raw_", n, ".rds", sep = ""))
  saveRDS(res_fft, paste(folder_path, "fft_", n, ".rds", sep = ""))
}
