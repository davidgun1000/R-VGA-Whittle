setwd("~/R-VGA-Whittle/AR1")

## R-VGA with Whittle likelihood (R-VGAW)?
rm(list = ls())

# library(stats)
# library(LSTS)
library(Matrix)
library(coda)
library(ggplot2)
library(tidyr)
library(mvtnorm)
source("./source/calculate_likelihood.R")
source("./source/run_rvgaw_ar1.R")
# source("./source/run_rvgae_ar1_archived.R")
# source("./source/run_mcmc_ar1.R")
# source("./source/run_vb_ar1.R")

result_directory <- "./results/"

## Flags
date <- "20230417" # 20240410 has phi = 0.9, 20230417 has phi = 0.7
regenerate_data <- F
save_data <- F

## R-VGA flags
rerun_rvgaw <- T
save_rvgaw_results <- F
use_tempering <- F
reorder_freq <- F
reorder_seed <- 2024
decreasing <- F
transform <- "arctanh"

## Model parameters 
phi <- 0.9
sigma_e <- 0.5
n <- 1000 # time series length

if (use_tempering) {
  n_temper <- 0.1 * n #floor(n/2) #10
  K <- 10
  temper_schedule <- rep(1/K, K)
  temper_info <- paste0("_temper", n_temper)
} else {
  n_temper <- 0
  temper_schedule <- NULL
  temper_info <- ""
}

if (reorder_freq) {
  reorder_info <- "_reordered"
} else {
  reorder_info <- ""
}

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

if (regenerate_data) {
  ## Generate AR(1) series
  x0 <- 1
  x <- c()
  x[1] <- x0
  set.seed(2023)
  for (t in 2:n) {
    x[t] <- phi * x[t-1] + rnorm(1, 0, sigma_e) 
  }
  rvgaw_data <- list(x = x, phi = phi, sigma_e = sigma_e)
  
  if (save_data) {
    saveRDS(rvgaw_data, file = paste0("./data/ar1_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  }
  
} else {
  rvgaw_data <- readRDS(file = paste0("./data/ar1_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  x <- rvgaw_data$x
  phi <- rvgaw_data$phi
  sigma_e <- rvgaw_data$sigma_e
}

plot(1:n, x, type = "l")

####################################################
##         R-VGA with Whittle likelihood          ##
####################################################

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_n", n, 
                         "_phi", phi_string, reorder_info, "_", date, ".rds")

S <- 200

# Priors: theta_phi ~ N(0, 1), theta_sigma ~ N(0, 0.5)
mu_0 <- c(1, -1) #log(sigma_e^2))
P_0 <- diag(c(1, 1))

# mu_0 <- 0
# P_0 <- 1

if (length(mu_0) > 1) {
  par(mfrow = c(2,1))
  test_theta_phi <- rnorm(10000, mu_0[1], P_0[1,1])
  phi_test <- tanh(test_theta_phi)
  hist(phi_test)
  test <- rnorm(10000, mu_0[2], P_0[2,2])
  sigma_e_test <- sqrt(exp(test))
  hist(sigma_e_test)
}

if (rerun_rvgaw) {
  
  rvgaw_results <- run_rvgaw_ar1(series = x, sigma_e = sigma_e, 
                                 prior_mean = mu_0, prior_var = P_0, 
                                 S = S, use_tempering = use_tempering,
                                 reorder_freq = reorder_freq,
                                 decreasing = decreasing,
                                 reorder_seed = reorder_seed,
                                 transform = transform)
  
  if (save_rvgaw_results) {
    saveRDS(rvgaw_results, rvgaw_filepath)
  }
  
} else {
  rvgaw_results <- readRDS(rvgaw_filepath)
}
rvgaw.post_samples <- rvgaw_results$post_samples


if (length(mu_0) > 1) {
  par(mfrow = c(2,1))
  plot(density(rvgaw.post_samples$phi), col = "red", lty = 2)
  abline(v = phi, lty = 2)
  
  plot(density(rvgaw.post_samples$sigma), col = "red", lty = 2)
  abline(v = sigma_e, lty = 2)
} else {
  plot(density(rvgaw.post_samples))
  abline(v = phi)
}
