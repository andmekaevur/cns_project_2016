rm(list=ls())
# install libraries if it is needed
if (!require("xgboost")) {
  install.packages("xgboost", repos="http://cran.rstudio.com/") 
}
library(xgboost)
if (!require("Matrix")) {
  install.packages("Matrix", repos="http://cran.rstudio.com/") 
}
library(Matrix)

# NB! might need to change PATH
setwd("~/git/cns_project_2016/R/Scripts")

# Paths to processed data and model folders + to result file
data_folder = "../Data/"
models_folder = "../Models/"
result_dir = "Results/"
dir.create(result_dir, showWarnings = FALSE)
result_file = paste0(result_dir, "xgb_res_", format(Sys.time(), "%m%d_%H%M"), ".csv")
print(paste("result will be in", result_file))

# seed value
seed = 123
set.seed(seed)

# set windows' width
timewindow = c(10, 20, 50, 75)

# subsample fraction (to minimize running time)
subsample_fraction = 1

# set fraction of data points to be used for experiment
training_fraction = 0.8

# number of rounds and threads for concurrent running
rounds_num = 120
threads_num = 16

# make separate folders for different experiments parameters
models_folder = paste0(models_folder, "xgb_", rounds_num, "_", subsample_fraction, "_", 
                       training_fraction, "/")
dir.create(models_folder, showWarnings = FALSE)

# turn off warning messages (to make it on change to 0)
options(warn=-1)

# calculating Mean Absolute Error
# simplified case (lists contains X and Y coordinates)
MAE = function(actual.x, actual.y, prediction.x, prediction.y) {
  mae = 1 / (2 * length(actual.x)) * (sum(abs(actual.x - prediction.x)) + sum(abs(actual.y - prediction.y)))
  return(mae)
}

# NB! currently script is working only for real number which means for raw data
# TODO: figure it out what to do with complex numbers
xgb_run = function (n, fft, seed=123) {
  cur_time = proc.time()
  set.seed(seed) 
  # create file if it does not exist yet
  if (!file.exists(result_file)) {
    header = data.frame(filename = "", rounds_number = rounds_num, window_size = 0, seed = 0, subsample_fraction = 0, 
                        training_fraction = 0, mae = 0, time_elapsed = "")
    write.table(header, file = result_file, row.names = FALSE, sep = ";")
  }
  
  # identify file name with processed data and read it
  if (fft) {
    filename = paste0("fft_", n)
  } else {
    filename = paste0("raw_", n)
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
  
  if(fft) {
    train.features = Re(train.features)
    test.features = Re(test.features)
  }
  
  # for xgboost data is needed to be in special kind of matrix
  dtrain.x <- xgb.DMatrix(data=train.features, label=train.x)
  dtrain.y <- xgb.DMatrix(data=train.features, label=train.y)

  print(paste("training started for", filename))
  
  print("X coordinate training ...")
  param <- list(objective  = "reg:linear",
                booster  = "gbtree",
                eval_metric  = "mae",
                eta = 0.1,
                gamma = 1,
                max_depth = 10,
                min_child_weight = 1, 
                max_delta_step = 0,
                subsample = 0.7,
                colsample_bytree = 1)
  xgb.x = xgboost(params = param, data = dtrain.x, label = train.x,
                     nrounds = rounds_num, nthread = threads_num)

  ### for estimation
  # xgb.cv(params = param, data = dtrain.x, early.stop.round=100, nrounds = rounds_num, 
  #        nfold = 5, nthread = 8, maximize = FALSE)
  
  print("Y coordinate training ...")
  xgb.y = xgboost(params = param, data = dtrain.y, label = train.y, 
                  nrounds = rounds_num, nthread = threads_num)
  
  # saving models
  model_x_file = paste0(models_folder, "xgb_x_", filename, "_", seed, "_", 
                        subsample_fraction, "_", training_fraction, ".rds")
  model_y_file = paste0(models_folder, "xgb_y_", filename, "_", seed, "_", 
                        subsample_fraction, "_", training_fraction, ".rds")
  
  saveRDS(xgb.x, file=model_x_file)
  saveRDS(xgb.y, file=model_y_file)
   
  print("prediction for X ...")
  xgb.x.pred = predict(xgb.x, test.features)
  print("prediction for Y ...")
  xgb.y.pred = predict(xgb.y, test.features)
  mae = MAE(test.x, test.y, xgb.x.pred, xgb.y.pred)

  time_elapsed = (proc.time() - cur_time)["elapsed"]
  print(time_elapsed)
  result = c(filename, rounds_num, n, seed, subsample_fraction, training_fraction, mae, time_elapsed)
  write.table(as.matrix(t(result)), file = result_file, sep=";",
              col.names = FALSE, row.names = FALSE, append = TRUE)
  cat("\n")
}

# iterates through all windows' sizes for raw data
for (n in timewindow) {
  xgb_run(n, FALSE)
}