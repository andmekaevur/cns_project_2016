rm(list = ls())

# NB! might need to change PATH
setwd("~/git/cns_project_2016/R/Scripts")

result_dir = "Results/"
result_file = paste0(result_dir, "xgb_tune_res_0111_1203_10_x.csv")
res = read.table(file = result_file, sep = ";", header = TRUE)

# sort hyperparamer search table by MAE value
res[order(res$bestMAE), ]
