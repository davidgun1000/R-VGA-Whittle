## R-VGA on SV model

setwd("~/R-VGA-Whittle/SV/")

library(mvtnorm)
library(Deriv)
library(tensorflow)
reticulate::use_condaenv("tf2.11", required = TRUE)
library(keras)

source("./source/run_rvgaw_sv_tf.R")
# source("./source/compute_kf_likelihood.R")
source("./source/compute_whittle_likelihood_sv.R")
# source("./source/update_sigma.R")

result_directory <- "./results/"

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

date <- "20230626"

## R-VGA flags
rerun_rvgaw <- T
save_rvgaw_results <- F
regenerate_data <- T
save_data <- F
use_tempering <- T
reorder_freq <- T
decreasing <- T
reorder_seed <- 2024

## Generate data
mu <- 0
phi <- 0.9
sigma_eta <- 0.7
sigma_eps <- 1
kappa <- 10
x1 <- rnorm(1, mu, sigma_eta^2 / (1 - phi^2))
n <- 1000
x <- c()
x[1] <- x1

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

## Generate data
if (regenerate_data) {
  for (t in 2:n) {
    x[t] <- phi * x[t-1] + sigma_eta * rnorm(1, 0, 1)
  }
  
  eps <- rnorm(n, 0, sigma_eps)
  y <- kappa * exp(x/2) * eps
  
  sv_data <- list(x = x, y = y, phi = phi, sigma_eta = sigma_eta, 
                  sigma_eps = sigma_eps, kappa = kappa)
  
  if (save_data) {
    saveRDS(sv_data, file = paste0("./data/sv_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  }
  
} else {
  print("Reading saved data...")
  sv_data <- readRDS(file = paste0("./data/sv_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  y <- sv_data$y
  x <- sv_data$x
  phi <- sv_data$phi
  sigma_eta <- sv_data$sigma_eta
  sigma_eps <- sv_data$sigma_eps
}

# par(mfrow = c(2,1))
# plot(y, type = "l")
# plot(x, type = "l")

## Test likelihood computation
# phi_grid <- seq(0.01, 0.99, length.out = 100)
# llh <- c()
# 
# for (k in 1:length(phi_grid)) {
#   params <- list(phi = phi_grid[k], sigma_eta = sigma_eta, sigma_xi = sqrt(pi^2/2))
#   llh[k] <- compute_whittle_likelihood_sv(y = y, params = params)
# }
# 
# plot(phi_grid, llh, type = "l")
# 
# test_eps <- rnorm(10000, 0, 1)
# test_xi <- log(test_eps^2)
# var(test_xi)

# Test exact likelihood
# phi_grid <- seq(0.01, 0.99, length.out = 100)
# llh <- c()
# 
# for (k in 1:length(phi_grid)) {
#   params_pf <- list(phi = phi_grid[k], sigma_eta = sigma_eta, sigma_eps = sigma_eps,
#                     sigma_xi = sqrt(pi^2/2))
#   pf_out <- particleFilter(y = y, N = 500, iniState = 0, param = params_pf)
#   llh[k] <- pf_out$log_likelihood
# }
# 
# plot(phi_grid, llh, type = "l")
# abline(v = phi_grid[which.max(llh)], col = "red",lty = 2)

# browser()

########################################
##                R-VGA               ##
########################################

S <- 500

if (use_tempering) {
  n_temper <- 10
  K <- 10
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

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_n", n, 
                         "_phi", phi_string, temper_info, reorder_info, "_", date, ".rds")

## Prior
prior_mean <- rep(0, 2) #c(atanh(phi), log(sigma_eta^2), log(pi^2/2)) #rep(0, 3)
prior_var <- diag(1, 2)

prior_theta <- rmvnorm(10000, prior_mean, prior_var)
prior_phi <- tanh(prior_theta[, 1])
prior_eta <- sqrt(exp(prior_theta[, 2]))
# prior_xi <- sqrt(exp(prior_theta[, 3]))
par(mfrow = c(3,1))
hist(prior_phi)
hist(prior_eta)
# hist(prior_xi)

if (rerun_rvgaw) {
  rvgaw_results <- run_rvgaw_sv(y = y, #sigma_eta = sigma_eta, sigma_eps = sigma_eps, 
                                prior_mean = prior_mean, prior_var = prior_var, 
                                deriv = "tf", 
                                S = S, use_tempering = use_tempering, 
                                reorder_freq = reorder_freq,
                                decreasing = decreasing, 
                                n_temper = n_temper,
                                temper_schedule = temper_schedule)
  
  if (save_rvgaw_results) {
    saveRDS(rvgaw_results, rvgaw_filepath)
  }
  
} else {
  rvgaw_results <- readRDS(rvgaw_filepath)
}

rvgaw.post_samples <- rvgaw_results$post_samples$phi
rvgaw.post_samples_phi <- rvgaw_results$post_samples$phi
rvgaw.post_samples_eta <- rvgaw_results$post_samples$sigma_eta
# rvgaw.post_samples_xi <- rvgaw_results$post_samples$sigma_xi


par(mfrow = c(1,3))
plot(density(rvgaw.post_samples_phi), main = "Posterior of phi")
abline(v = phi, lty = 2)

plot(density(rvgaw.post_samples_eta), main = "Posterior of sigma_eta")
abline(v = sigma_eta, lty = 2)

# plot(density(rvgaw.post_samples_xi), main = "Posterior of sigma_xi")
# abline(v = sqrt(pi^2/2), lty = 2)

## Trajectories
mu_phi <- sapply(rvgaw_results$mu, function(x) x[1])
mu_eta <- sapply(rvgaw_results$mu, function(x) x[2])
# mu_xi <- sapply(rvgaw_results$mu, function(x) x[3])

par(mfrow = c(1, 2))
plot(tanh(mu_phi), type = "l", main = "Trajectory of phi")
abline(h = phi, lty = 2)

plot(sqrt(exp(mu_eta)), type = "l", main = "Trajectory of sigma_eta")
abline(h = sigma_eta, lty = 2)

# plot(sqrt(exp(mu_xi)), type = "l", main = "Trajectory of sigma_xi")
# abline(h = sqrt(pi^2/2), lty = 2)

