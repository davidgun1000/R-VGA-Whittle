setwd("~/R-VGA-Whittle/LGSS/")
rm(list = ls())

library(tensorflow)
reticulate::use_condaenv("tf2.11", required = TRUE)
library(keras)
library(mvtnorm)
library(coda)
library(Deriv)

source("./source/run_rvgaw_lgss_tf.R")
# source("./source/compute_kf_likelihood.R")
source("./source/compute_whittle_likelihood_lb.R")
# source("./source/update_sigma.R")


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


## Flags
date <- "20230525"
regenerate_data <- T
save_data <- T
use_tempering <- T
reorder_freq <- T
decreasing <- T
reorder_seed <- 2024

## True parameters
sigma_eps <- 0.5 # measurement error var
sigma_eta <- 0.7 # process error var
phi <- 0.9

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

# Generate true process x_1:T
n <- 1000
# times <- seq(0, 1, length.out = iters)

if (regenerate_data) {
  x <- c()
  x[1] <- 1
  set.seed(2023)
  for (t in 2:n) {
    x[t] <- phi * x[t-1] + rnorm(1, 0, sigma_eta)
  }
  
  # Generate observations y_1:T
  y <- x + rnorm(n, 0, sigma_eps)
  
  ## Plot true process and observations
  # par(mfrow = c(1, 1))
  # plot(x, type = "l", main = "True process")
  # points(y, col = "cyan")
  
  lgss_data <- list(x = x, y = y, phi = phi, sigma_eps = sigma_eps, sigma_eta = sigma_eta)
  if (save_data) {
    saveRDS(lgss_data, file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  }
} else {
  lgss_data <- readRDS(file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  y <- lgss_data$y
  x <- lgss_data$x
  phi <- lgss_data$phi
  sigma_eps <- lgss_data$sigma_eps
  sigma_eta <- lgss_data$sigma_eta
}


## R-VGA with Whittle likelihood on LGSS

if (use_tempering) {
  n_temper <- 10
  K <- 10
  temper_schedule <- rep(1/K, K)
}

## Prior
prior_mean <- rep(0, 3)
prior_var <- diag(1, 3)

rvgaw_results <- run_rvgaw_lgss(y = y, #sigma_eta = sigma_eta, sigma_eps = sigma_eps, 
                                prior_mean = prior_mean, prior_var = prior_var, 
                                deriv = "tf", 
                                S = 200, use_tempering = use_tempering, 
                                reorder_freq = reorder_freq,
                                decreasing = decreasing, 
                                n_temper = n_temper,
                                temper_schedule = temper_schedule)

# if (save_rvgaw_results) {
#   saveRDS(rvgaw_results, rvgaw_filepath)
# }

rvgaw.post_samples <- rvgaw_results$post_samples$phi
rvgaw.post_samples_phi <- rvgaw_results$post_samples$phi
rvgaw.post_samples_eta <- rvgaw_results$post_samples$sigma_eta
rvgaw.post_samples_eps <- rvgaw_results$post_samples$sigma_eps


par(mfrow = c(1,3))
plot(density(rvgaw.post_samples_phi), main = "Posterior of phi")
abline(v = phi, lty = 2)

plot(density(rvgaw.post_samples_eta), main = "Posterior of sigma_eta")
abline(v = sigma_eta, lty = 2)

plot(density(rvgaw.post_samples_eps), main = "Posterior of sigma_eps")
abline(v = sigma_eps, lty = 2)

## LB
plot(rvgaw_results$lower_bound, type = "l")
