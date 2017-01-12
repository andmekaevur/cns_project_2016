## PREPARE DATA FOR CONDUCTING AN EXPERIMENT AND STORE IT ##  

rm(list = ls())
# install libraries if is needed
if (!require("R.matlab")) {
  install.packages("R.matlab", repos="http://cran.rstudio.com/") 
}
if (!require("randomForest")) {
  install.packages("randomForest", repos="http://cran.rstudio.com/") 
}
library("R.matlab")
library(randomForest)

# NB! might need to change PATH
setwd("~/git/cns_project_2016/R/Scripts")

# Reading given data
# X_raw:
# - rows are EEG signals for different channels 
# - columns are measurements for different experiments with 20 ms interval
X_raw = readMat("./../../R2198_20ms.mat")[["mm"]]
# Y_raw: 
# - columns are coordinates of rat"s position (x, y) 
Y_raw = read.table("./../../R2198_locations.dat")

# prepare data sets for condacting an experiment
# Experiment: based on timewindow [t - n; t + n] predict coordinates (x, y) for experiment #t 
# feature vector for experiment #t is glued togeather timeslices [t - n; t + n] of different channels
# IN: n (int) - size of timewindow; fft (bool) - whether to apply fft or not before concatenation of channel's timeslices
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

# create directory for saving data (if there is no such)
data_dir = "../Data/"
dir.create(data_dir, showWarnings = FALSE)

# Save prepared datasets 
timewindow = c(10, 20, 50, 75)
for (n in timewindow) {
  res_raw = prepare_datasets(n, FALSE)
  res_fft = prepare_datasets(n, TRUE)
  saveRDS(res_raw, paste(data_dir, "raw_", n, ".rds", sep = ""))
  saveRDS(res_fft, paste(data_dir, "fft_", n, ".rds", sep = ""))
}
