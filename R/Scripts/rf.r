## READ PROCESSED DATA AND APPLY RANDOM FOREST ##  

rm(list = ls())
# install libraries if it is needed
if (!require("randomForest")) {
  install.packages("randomForest", repos="http://cran.rstudio.com/") 
}
library(randomForest)
# library(foreach)
# library(doSNOW)

# NB! might need to change PATH
setwd("~/git/cns_project_2016/R/Scripts")

# Paths to processed data and model folders + to result file
data_folder = "../Data/"
models_folder = "../Models/"
result_dir = "Results/"
dir.create(result_dir, showWarnings = FALSE)
result_file = paste0("rf_res_", format(Sys.time(), "%m%d_%H%M"), ".csv")

# seed value
seed = 123
set.seed(seed)

# set windows' width
timewindow = c(10, 20, 50, 75)

# subsample fraction (to minimize running time)
subsample_fraction = 0.5

# set fraction of data points to be used for experiment
training_fraction = 0.8

# set num of nodes for parallelisation 
nodes = 6

# necessary for concurrent running
# registerDoSNOW(makeCluster(nodes, type="SOCK"))

# set trees num for random forest
trees_num = 10

# make separate folders for different experiments parameters
models_folder = paste0(models_folder, trees_num, "_", subsample_fraction, "_", training_fraction, "/")
dir.create(models_folder, showWarnings = FALSE)

# turn off warning messages (to make it on change to 0)
options(warn=-1)

# calculating Mean Absolute Error
# simplified case (lists contains X and Y coordinates)
MAE = function(actual.x, actual.y, prediction.x, prediction.y) {
  mae = 1 / (2 * length(actual.x)) * (sum(abs(actual.x - prediction.x)) + sum(abs(actual.y - prediction.y)))
  return(mae)
}

## LEARNING MODELS ##
# method for learning and saving RF model for one variant of processed data
# models for x and y coordinates are different
# IN: n (int) - length of timewindow ([t-n; t+n])
# fft (bool) - whether to use data with fft applied
# seed for running an experiment
rf_run = function (n, fft, seed=123) {
  cur_time = proc.time()
  set.seed(seed) 
  # create file if it does not exist yet
  if (!file.exists(result_file)) {
    header = data.frame(filename = "", preprocessing = "", window_size = 0, trees_num = 0, seed = 0,
                        subsample_fraction = 0, training_fraction = 0, mae = 0, time_elapsed = "")
    write.table(header, file = result_file, row.names = FALSE, sep = ";")
  }
  
  # identify file name with processed data and read it
  if (fft) {
    filename = paste0("fft_", n)
    preprocessing = "fft"
  } else {
    filename = paste0("raw_", n)
    preprocessing = "raw"
  }
  data = readRDS(file=paste0(data_folder, filename, ".rds"))
  
  # subsample data to reduce calculations
  total_num_of_experiments = nrow(data$X)
  subsample_size = floor(total_num_of_experiments * subsample_fraction)
  subsample_ind = sample(seq_len(total_num_of_experiments), size = subsample_size)
  data$X = data$X[subsample_ind, ]
  data$Y = data$Y[subsample_ind, ]
  
  # find indexes of training rows 
  num_of_experiments = nrow(data$X)
  training_size = floor(training_fraction * num_of_experiments)
  train_ind = sample(seq_len(num_of_experiments), size = training_size)
  
  # split dataset into training and test
  train.features = data$X[train_ind, ]
  test.features = data$X[-train_ind, ]
  
  train.x = data$Y[train_ind, 1]
  test.x = data$Y[-train_ind, 1]
  
  train.y = data$Y[train_ind, 2]
  test.y = as.data.frame(data$Y[-train_ind, 2])
  
  # run rf (based on target type it will be RF for regression)
  print(paste("training started for", filename))
  
  print("X coordinate training ...")
  rf.x = randomForest(x=train.features, y = train.x, ntree = trees_num, importance = TRUE)
  # NB! there are some problems with running in concurrently, however, the code should work
  # rf.x = foreach(trees = rep(trees_num/nodes, nodes), .combine = combine, .packages = "randomForest") %dopar%
  #   randomForest(x=train.features, y = train.x, ntree = trees, importance = TRUE)

  model_x_file = paste0(models_folder, "rf_x_", filename, "_", trees_num, "_", seed, "_", subsample_fraction, ".rds")
  saveRDS(rf.x, file=model_x_file)
  
  print("Y coordinate training ...")
  rf.y = randomForest(x=train.features, y = train.y, ntree = trees_num, importance = TRUE)
  # rf.y = foreach(trees = rep(trees_num/nodes, nodes), .combine = combine, .packages = "randomForest") %dopar%
  #   randomForest(x=train.features, y = train.y, ntree = trees, importance = TRUE)
  
  model_y_file = paste0(models_folder, "rf_y_", filename, "_", trees_num, "_", seed, "_", subsample_fraction, ".rds")
  saveRDS(rf.y, file=model_y_file)
  
  print("prediction for X ...")
  rf.x.pred = predict(rf.x, test.features)
  print("prediction for Y ...")
  rf.y.pred = predict(rf.y, test.features)
  mae = MAE(test.x, test.y, rf.x.pred, rf.y.pred)
  
  time_elapsed = (proc.time() - cur_time)["elapsed"]
  print(time_elapsed)
  result = c(filename, preprocessing, n, trees_num, seed, subsample_fraction, training_fraction, mae, time_elapsed)
  write.table(as.matrix(t(result)), file = result_file, sep=";", 
              col.names = FALSE, row.names = FALSE, append = TRUE)
  cat("\n")
}

# iterates through all window sizes
# for raw data
for (n in timewindow) {
  rf_run(n, FALSE)
}

# for preprocessed data by FFT
for (n in timewindow) {
  rf_run(n, TRUE)
}
