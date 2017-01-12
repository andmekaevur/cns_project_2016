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
data_dir = "../Data/"
models_dir = "../Models/"
result_dir = "Results/"
dir.create(result_dir, showWarnings = FALSE)
# it will be added later ending _x.csv or _y.csv
result_file = paste0(result_dir, "xgb_tune_res_", format(Sys.time(), "%m%d_%H%M"))
print(paste("result will be in", result_file))

# seed value
seed = 123
set.seed(seed)

# nodes number for concurrent running
nodes = 8

# set windows' width
timewindow = c(10, 20, 50, 75)

# subsample fraction (to minimize running time)
subsample_fraction = 0.5

# set fraction of data points to be used for experiment
training_fraction = 0.8

# hyper parameters searh for tuning
folds = c(4)
eta = c(0.3, 0.1)
lambda = c(0, 1)
rounds_num = 200

# make separate folders for different experiments parameters
models_dir = paste0(models_dir, "xgb_tune_", subsample_fraction, "_", 
                       training_fraction, "/")
dir.create(models_dir, showWarnings = FALSE)

# turn off warning messages (to make it on change to 0)
options(warn=-1)

# NB! currently script is working only for real number which means for raw data
# TODO: figure it out what to do with complex numbers
xgb_tune = function (n, seed=123) {
  cur_time = proc.time()
  set.seed(seed) 
  
  data_filename = paste0("raw_", n)
  data = readRDS(file=paste0(data_dir, data_filename, ".rds"))
  
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
  
  # for xgboost data is needed to be in special kind of matrix
  dtrain.x <- xgb.DMatrix(data=train.features, label=train.x)
  dtrain.y <- xgb.DMatrix(data=train.features, label=train.y)
  
  # combine all parameters into data frame
  N = expand.grid(folds, eta, lambda)
  results = as.data.frame(N)
  names(results)=c("folds", "eta", "lambda")
  results$bestMAE = -1
  results$bestRound = -1
  
  print(paste("tuning started for ", data_filename))
  print("X coordinate ...")
  # main loop for estimating different parameters for X coordinate
  for (i in 1:nrow(N)) {
    print(i)
    param <- list(objective  = "reg:linear",
                  booster  = "gblinear",
                  eval_metric  = "mae",
                  eta = N[i,2],
                  lambda = N[i,3])
    # use early stoping to speed up the process
    CV_current = xgb.cv(params = param, data = dtrain.x, early.stop.round=100, nrounds = rounds_num, 
                        nfold = N[i,1], nthread = nodes, maximize = FALSE)
    CV_current$MAE = CV_current$evaluation_log[, "test_mae_mean"]
    results$bestRound[i] = CV_current$best_iteration
    results$bestMAE[i] = CV_current$MAE[CV_current$best_iteration][[1]]
    
    write.table(results, file = paste0(result_file, "_x.csv"), quote = FALSE,
                append = FALSE, col.names=TRUE, row.names=FALSE, sep=";")
  }
  
  print("Y coordinate ...")
  # main loop for estimating different parameters for X coordinate
  for (i in 1:nrow(N)) {
    print(i)
    param <- list(objective  = "reg:linear",
                  booster  = "gblinear",
                  eval_metric  = "mae",
                  eta = N[i,2],
                  lambda = N[i,3])
    # use early stoping to speed up the process
    CV_current = xgb.cv(params = param, data = dtrain.y, early.stop.round=100, nrounds = rounds_num, 
                        nfold = N[i,1], nthread = nodes, maximize = FALSE)
    CV_current$MAE = CV_current$evaluation_log[, "test_mae_mean"]
    results$bestRound[i] = CV_current$best_iteration
    results$bestMAE[i] = CV_current$MAE[CV_current$best_iteration][[1]]
    
    write.table(results, file = paste0(result_file, "_y.csv"), quote = FALSE,
                append = FALSE, col.names=TRUE, row.names=FALSE, sep=";")
  }

  time_elapsed = (proc.time() - cur_time)["elapsed"]
  print(time_elapsed)
  cat("\n")
}

for (n in timewindow) {
  xgb_tune(n)
}
