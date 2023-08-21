## Trivariate SV model

## R-VGAW on the SV model

## MCMC on multivariate SV model
rm(list = ls())
setwd("~/R-VGA-Whittle/Multi_SV/")

library("coda")
library("mvtnorm")
library("astsa")
library("cmdstanr")
# library("expm")
library("stcos")
library(tensorflow)
reticulate::use_condaenv("tf2.11", required = TRUE)
library(keras)

source("./source/run_rvgaw_multi_sv.R")
source("./source/run_mcmc_multi_sv.R")
source("./source/compute_whittle_likelihood_multi_sv.R")
source("./source/run_mcmc_sv.R")
source("./source/compute_whittle_likelihood_sv.R")
source("./source/construct_prior.R")
source("./source/map_functions.R")
# source("./archived/compute_partial_whittle_likelihood.R")
source("./source/compute_grad_hessian.R")

# List physical devices
gpus <- tf$config$experimental$list_physical_devices('GPU')

if (length(gpus) > 0) {
  tryCatch({
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    tf$config$experimental$set_virtual_device_configuration(
      gpus[[1]],
      list(tf$config$experimental$VirtualDeviceConfiguration(memory_limit=4096))
    )
    
    logical_gpus <- tf$config$experimental$list_logical_devices('GPU')
    
    print(paste0(length(gpus), " Physical GPUs,", length(logical_gpus), " Logical GPUs"))
  }, error = function(e) {
    # Virtual devices must be set before GPUs have been initialized
    print(e)
  })
}

result_directory <- "./results/"

## Flags
date <- "20230814"
regenerate_data <- T
save_data <- F
use_cholesky <- F # use lower Cholesky factor to parameterise Sigma_eta


#######################
##   Generate data   ##
#######################

dataset <- "0"

Tfin <- 1000

if (regenerate_data) {
  
  Phi <- diag(c(0.7, 0.8, 0.9))
  
  sigma_eta1 <- 0.9#4
  sigma_eta2 <- 1.5 #1.5
  sigma_eta3 <- 1.2 #1.5
  
  Sigma_eta <- diag(c(sigma_eta1, sigma_eta2, sigma_eta3))
  
  d <- dim(Phi)[1] # time series dimension
  
  # sigma_eps1 <- 1 #0.01
  # sigma_eps2 <- 1 #0.02
  # 
  # Sigma_eps <- diag(c(sigma_eps1, sigma_eps2))
  Sigma_eps <- diag(d)
  
  x1 <- rmvnorm(1, rep(0, d), compute_autocov_VAR1(Phi, Sigma_eta))
  X <- matrix(NA, nrow = length(x1), ncol = Tfin+1) # x_0:T
  X[, 1] <- x1
  Y <- matrix(NA, nrow = length(x1), ncol = Tfin) # y_1:T
  # Y[, 1] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
  set.seed(2022)
  for (t in 1:Tfin) {
    X[, t+1] <- Phi %*% X[, t] + t(rmvnorm(1, rep(0, d), Sigma_eta))
    V <- diag(exp(X[, t+1]/2))
    Y[, t] <- V %*% t(rmvnorm(1, rep(0, d), Sigma_eps))
  }
  
  multi_sv_data <- list(X = X, Y = Y, Phi = Phi, Sigma_eta = Sigma_eta, 
                        Sigma_eps = Sigma_eps)
  
  if (save_data) {
    saveRDS(multi_sv_data, file = paste0("./data/multi_sv_data_Tfin", Tfin, "_", date, "_", dataset, ".rds"))
  }
} else {
  multi_sv_data <- readRDS(file = paste0("./data/multi_sv_data_Tfin", Tfin, "_", date, "_", dataset, ".rds"))
  
  X <- multi_sv_data$X
  Y <- multi_sv_data$Y
  Phi <- multi_sv_data$Phi
  Sigma_eta <- multi_sv_data$Sigma_eta
  Sigma_eps <- multi_sv_data$Sigma_eps
}

par(mfrow = c(d,1))
plot(X[1, ], type = "l")
plot(X[2, ], type = "l")
plot(X[3, ], type = "l")

plot(Y[1, ], type = "l")
plot(Y[2, ], type = "l")
plot(Y[3, ], type = "l")


