rm(list=ls())
# install libraries if it is needed
if (!require("ggplot2")) {
  install.packages("ggplot2", repos="http://cran.rstudio.com/") 
}
library(ggplot2)

# NB! might need to change PATH
setwd("~/git/cns_project_2016/R/Scripts")

result_dir = "Results/"
results_file = paste0(result_dir, "rf.csv")

data = read.table(file = results_file, header = TRUE, sep = ";")
data.raw = data[which(data$preprocessing == "raw"), ]
data.fft = data[which(data$preprocessing == "fft"), ]

ggplot(data.raw, aes(x = data.raw$window_size)) + 
  geom_line(aes(y = data.raw$mae, colour = "Raw data")) + 
  geom_line(aes(y = data.fft$mae, colour = "FFT"))