## R-VGAW on the SV model

## MCMC on multivariate SV model
rm(list = ls())
setwd("~/R-VGA-Whittle/Multi_SV/")

library("coda")
library("mvtnorm")
library("astsa")
# library("expm")
library("stcos")
library(tensorflow)
reticulate::use_condaenv("tf2.11", required = TRUE)
library(keras)

# source("./source/compute_whittle_likelihood_multi_sv.R")
# source("./source/run_mcmc_multi_sv.R")
# source("./source/compute_whittle_likelihood_multi_sv.R")

source("./source/run_rvgaw_multi_sv.R")
source("./source/construct_prior.R")
source("./source/map_functions.R")
source("./source/compute_partial_whittle_likelihood.R")
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

## Result directory
result_directory <- "./results/"
date <- "20230807"

dataset <- "_0"

## Flags
regenerate_data <- T
save_data <- F

rerun_rvgaw <- T
save_rvgaw_results <- F

## R-VGA flags
use_tempering <- T
reorder_freq <- F
decreasing <- T

#######################
##   Generate data   ##
#######################

Tfin <- 1000

if (regenerate_data) {
  phi11 <- 0.9
  phi12 <- 0#.1
  phi21 <- 0#.2
  phi22 <- 0.7
  
  Phi <- matrix(c(phi11, phi12, phi21, phi22), 2, 2, byrow = T)
  
  sigma_eta1 <- 0.5
  sigma_eta2 <- 2.5
  Sigma_eta <- diag(c(sigma_eta1, sigma_eta2))
  
  # Sigma_eta <- matrix(c(0.6, 0.1, 0.1, 1.2), 2, 2, byrow = T)
  
  sigma_eps1 <- 1 #0.01
  sigma_eps2 <- 1 #0.02
  
  Sigma_eps <- diag(c(sigma_eps1, sigma_eps2))
  
  
  x1 <- c(0, 0)
  X <- matrix(NA, nrow = length(x1), ncol = Tfin+1) # x_0:T
  X[, 1] <- x1
  Y <- matrix(NA, nrow = length(x1), ncol = Tfin) # y_1:T
  # Y[, 1] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
  for (t in 1:Tfin) {
    X[, t+1] <- Phi %*% X[, t] + t(rmvnorm(1, c(0, 0), Sigma_eta))
    V <- diag(exp(X[, t+1]/2))
    Y[, t] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
  }
  
  multi_sv_data <- list(X = X, Y = Y, Phi = Phi, Sigma_eta = Sigma_eta, 
                        Sigma_eps = Sigma_eps)
  
  
  if (save_data) {
    saveRDS(multi_sv_data, file = paste0("./data/multi_sv_data_T", Tfin, "_", date, dataset, ".rds"))
  }
} else {
  multi_sv_data <- readRDS(file = paste0("./data/multi_sv_data_T", Tfin, "_", date, dataset, ".rds"))
  Y <- multi_sv_data$Y
  X <- multi_sv_data$X
  Phi <- multi_sv_data$Phi
  Sigma_eta <- multi_sv_data$Sigma_eta
  Sigma_eps <- multi_sv_data$Sigma_eps
}


# par(mfrow = c(2,1))
# plot(X[1, ], type = "l")
# plot(X[2, ], type = "l")
# 
# plot(Y[1, ], type = "l")
# plot(Y[2, ], type = "l")
################################
##    R-VGAW implementation   ##
################################

if (use_tempering) {
  n_temper <- 10
  K <- 100
  temper_schedule <- rep(1/K, K)
  temper_info <- paste0("_temper", n_temper)
} else {
  temper_info <- ""
}

if (reorder_freq) {
  reorder_info <- "_reorder"
} else {
  reorder_info <- ""
}

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_T", Tfin, 
                         temper_info, reorder_info, "_", date, dataset, ".rds")

## Construct initial distribution/prior
prior <- construct_prior(data = Y)
prior_mean <- prior$prior_mean
prior_var <- prior$prior_var

param_dim <- length(prior_mean)

################ R-VGA starts here #################

S <- 100L

if (rerun_rvgaw) {
  rvgaw_results <- run_rvgaw_multi_sv(Y, prior_mean, prior_var, S, 
                                      use_tempering = use_tempering,
                                      reorder_freq = reorder_freq,
                                      decreasing = decreasing,
                                      temper_schedule = temper_schedule)

  if (save_rvgaw_results) {
    saveRDS(rvgaw_results, rvgaw_filepath)
  }  
} else {
  rvgaw_results <- readRDS(rvgaw_filepath)
}

rvgaw.post_samples_Phi <- rvgaw_results$post_samples$Phi
rvgaw.post_samples_Sigma_eta <- rvgaw_results$post_samples$Sigma_eta

rvgaw.post_samples_phi_11 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[1,1]))
rvgaw.post_samples_phi_12 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[1,2]))
rvgaw.post_samples_phi_21 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[2,1]))
rvgaw.post_samples_phi_22 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[2,2]))

rvgaw.post_samples_sigma_eta_11 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[1,1]))
rvgaw.post_samples_sigma_eta_22 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[2,2]))

par(mfrow = c(3,2))
plot(density(rvgaw.post_samples_phi_11), col = "blue", main = "phi_11")
abline(v = Phi[1,1], lty = 2)
plot(density(rvgaw.post_samples_phi_12), col = "blue", main = "phi_12")
abline(v = Phi[1,2], lty = 2)
plot(density(rvgaw.post_samples_phi_21), col = "blue", main = "phi_21")
abline(v = Phi[2,1], lty = 2)
plot(density(rvgaw.post_samples_phi_22), col = "blue", main = "phi_22")
abline(v = Phi[2,2], lty = 2)

plot(density(rvgaw.post_samples_sigma_eta_11), col = "blue", main = "sigma_eta_11")
abline(v = Sigma_eta[1,1], lty = 2)
plot(density(rvgaw.post_samples_sigma_eta_22), col = "blue", main = "sigma_eta_22")
abline(v = Sigma_eta[2,2], lty = 2)

### Trajectories
A_list <- lapply(rvgaw_results$mu, function(x) matrix(x[1:4], 2, 2, byrow = T))
Sigma_eta_list <- lapply(rvgaw_results$mu, function(x) diag(exp(x[5:6])))

Phi_list <- mapply(backward_map, A_list, Sigma_eta_list, SIMPLIFY = F)  
  
phi11_traject <- unlist(lapply(Phi_list, function(x) x[1,1]))
phi12_traject <- unlist(lapply(Phi_list, function(x) x[1,2]))
phi21_traject <- unlist(lapply(Phi_list, function(x) x[2,1]))
phi22_traject <- unlist(lapply(Phi_list, function(x) x[2,2]))

sigma11_traject <- unlist(lapply(Sigma_eta_list, function(x) x[1,1]))
sigma22_traject <- unlist(lapply(Sigma_eta_list, function(x) x[2,2]))

par(mfrow = c(3,2))
plot(phi11_traject, type = "l")
abline(h = Phi[1,1], lty = 2, col = "red")
plot(phi12_traject, type = "l")
abline(h = Phi[1,2], lty = 2, col = "red")
plot(phi21_traject, type = "l")
abline(h = Phi[2,1], lty = 2, col = "red")
plot(phi22_traject, type = "l")
abline(h = Phi[2,2], lty = 2, col = "red")
plot(sigma11_traject, type = "l")
abline(h = Sigma_eta[1,1], lty = 2, col = "red")
plot(sigma22_traject, type = "l")
abline(h = Sigma_eta[2,2], lty = 2, col = "red")
