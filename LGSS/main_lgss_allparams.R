setwd("~/R-VGA-Whittle/LGSS/")
rm(list = ls())

library(mvtnorm)
library(coda)
library(Deriv)
library(tensorflow)
reticulate::use_condaenv("tf2.11", required = TRUE)
library(keras)

source("./source/run_rvgaw_lgss_tf.R")
source("./source/run_mcmc_lgss_allparams.R")
source("./source/compute_kf_likelihood.R")
source("./source/compute_whittle_likelihood_lgss.R")
# source("./source/compute_whittle_likelihood_lb.R")
source("./source/update_sigma.R")
source("./source/run_hmc_lgss.R")
################## Some code to limit tensorflow memory usage ##################

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

################## End of code to limit tensorflow memory usage ##################

## Result directory
result_directory <- "./results/"

## Flags
date <- "20230525"
regenerate_data <- T
save_data <- F

rerun_rvgaw <- T
rerun_mcmcw <- T
# rerun_mcmce <- F
rerun_hmc <- T
save_rvgaw_results <- F
save_mcmcw_results <- F
# save_mcmce_results <- F
save_hmc_results <- F

## R-VGA flags
use_tempering <- T
reorder_freq <- T
decreasing <- T
transform <- "arctanh"

## MCMC flags
adapt_proposal <- T

## True parameters
sigma_eps <- 0.5 # measurement error var
sigma_eta <- 0.7 # process error var
phi <- 0.8

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

n <- 1000

if (regenerate_data) {
  print("Generating data...")
  
  # Generate true process x_0:T
  x <- c()
  x0 <- rnorm(1, 0, sqrt(sigma_eta^2 / (1-phi^2)))
  x[1] <- x0
  set.seed(2023)
  for (t in 2:(n+1)) {
    x[t] <- phi * x[t-1] + rnorm(1, 0, sigma_eta)
  }
  
  # Generate observations y_1:T
  y <- x[2:(n+1)] + rnorm(n, 0, sigma_eps)
  
  ## Plot true process and observations
  # par(mfrow = c(1, 1))
  plot(x, type = "l", main = "True process")
  # points(y, col = "cyan")
  
  lgss_data <- list(x = x, y = y, phi = phi, sigma_eps = sigma_eps, sigma_eta = sigma_eta)
  if (save_data) {
    saveRDS(lgss_data, file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  }
} else {
  print("Reading saved data...")
  lgss_data <- readRDS(file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  y <- lgss_data$y
  x <- lgss_data$x
  phi <- lgss_data$phi
  sigma_eps <- lgss_data$sigma_eps
  sigma_eta <- lgss_data$sigma_eta
}

## MCMC settings
burn_in <- 5000
n_post_samples <- 10000
MCMC_iters <- n_post_samples + burn_in # Number of MCMC iterations

## Prior
prior_mean <- rep(0, 3)
prior_var <- diag(1, 3)

## Initial state mean and variance for the KF
state_ini_mean <- 0
state_ini_var <- 1

## Test the likelihood computation by plotting likelihood surface over a grid of parameter values
# phi_grid <- seq(-1, 1, length.out = 200)
# likelihood_whittle <- c()
# likelihood_exact <- c()
# 
# for (i in 1:length(phi_grid)) {
#   params_list <- list(phi = phi_grid[i], sigma_eta = sigma_eta, sigma_eps = sigma_eps)
#   # if (use_whittle_likelihood) {
#   likelihood_whittle[i] <- compute_whittle_likelihood_lgss(y = y, params = params_list)
#   # } else {
#   kf_out <- compute_kf_likelihood(state_prior_mean = state_ini_mean,
#                                   state_prior_var = state_ini_var,
#                                   iters = length(y), observations = y,
#                                   params = params_list)
#   
#   likelihood_exact[i] <- kf_out$log_likelihood
#   # }
# }
# par(mfrow = c(2,1))
# margin <- 20
# plot_range <- (which.max(likelihood_exact) - margin):(which.max(likelihood_exact) + margin)
# plot(phi_grid[plot_range], likelihood_exact[plot_range], type = "l",
#      xlab = "phi", ylab = "log likelihood", main = paste0("Exact likelihood (n = ", n, ")"))
# legend("topleft", legend = c("true value", "arg max llh"),
#        col = c("black", "red"), lty = 2, cex = 0.5)
# abline(v = phi_grid[which.max(likelihood_exact)], lty = 1, col = "red")
# abline(v = phi, lty = 2)
# 
# plot(phi_grid[plot_range], likelihood_whittle[plot_range], type = "l",
#      xlab = "phi", ylab = "log likelihood", main = paste0("Whittle likelihood (n = ", n, ")"))
# legend("topleft", legend = c("true value", "arg max llh"),
#        col = c("black", "red"), lty = 2, cex = 0.5)
# abline(v = phi_grid[which.max(likelihood_whittle)], lty = 1, col = "red")
# abline(v = phi, lty = 2)

## Sample from prior to see if values of phi are reasonable
# theta_samples <- rnorm(10000, prior_mean, prior_var)
# if (transform == "arctanh") {
#   phi_samples <- tanh(theta_samples)
# } else {
#   phi_samples <- exp(theta_samples) / (1 + exp(theta_samples))
# }
# 
# hist(phi_samples, main = "Samples from the prior of phi")

################################################################################
##                      R-VGA with Whittle likelihood                         ##
################################################################################

if (use_tempering) {
  n_temper <- 10 #0.1 * n
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

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_", transform, "_n", n,
                         "_phi", phi_string, temper_info, reorder_info, "_", date, ".rds")

if (rerun_rvgaw) {
  rvgaw_results <- run_rvgaw_lgss(y = y, #sigma_eta = sigma_eta, sigma_eps = sigma_eps, 
                                  prior_mean = prior_mean, prior_var = prior_var, 
                                  deriv = "tf", 
                                  S = 100, use_tempering = use_tempering, 
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

rvgaw.post_samples_phi <- rvgaw_results$post_samples$phi
rvgaw.post_samples_eta <- rvgaw_results$post_samples$sigma_eta
rvgaw.post_samples_eps <- rvgaw_results$post_samples$sigma_eps

# ################################################################################
# ##                        MCMC with exact likelihood                          ##   
# ################################################################################
# 
# mcmce_filepath <- paste0(result_directory, "mcmc_exact_results_n", n,
#                          "_phi", phi_string, "_", date, ".rds")
# 
# if (rerun_mcmce) {
#   mcmce_results <- run_mcmc_lgss(y, #sigma_eta, sigma_eps,
#                                  iters = MCMC_iters, burn_in = burn_in,
#                                  prior_mean = prior_mean, prior_var = prior_var,
#                                  state_ini_mean = state_ini_mean, state_ini_var = state_ini_var,
#                                  adapt_proposal = T, use_whittle_likelihood = F)
# 
#   if (save_mcmce_results) {
#     saveRDS(mcmce_results, mcmce_filepath)
#   }
# 
# } else {
#   mcmce_results <- readRDS(mcmce_filepath)
# }
# 
# mcmce.post_samples_phi <- as.mcmc(mcmce_results$post_samples$phi[-(1:burn_in)])
# mcmce.post_samples_eta <- as.mcmc(mcmce_results$post_samples$sigma_eta[-(1:burn_in)])
# mcmce.post_samples_eps <- as.mcmc(mcmce_results$post_samples$sigma_eps[-(1:burn_in)])

################################################################################
##                       MCMC with Whittle likelihood                         ##
################################################################################

mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_results_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

if (rerun_mcmcw) {
  mcmcw_results <- run_mcmc_lgss(y, #sigma_eta, sigma_eps, 
                                 iters = MCMC_iters, burn_in = burn_in,
                                 prior_mean = prior_mean, prior_var = prior_var,  
                                 state_ini_mean = state_ini_mean, state_ini_var = state_ini_var,
                                 adapt_proposal = T, use_whittle_likelihood = T)
  
  if (save_mcmcw_results) {
    saveRDS(mcmcw_results, mcmcw_filepath)
  }
} else {
  mcmcw_results <- readRDS(mcmcw_filepath)
}

mcmcw.post_samples_phi <- as.mcmc(mcmcw_results$post_samples$phi[-(1:burn_in)])
mcmcw.post_samples_eta <- as.mcmc(mcmcw_results$post_samples$sigma_eta[-(1:burn_in)])
mcmcw.post_samples_eps <- as.mcmc(mcmcw_results$post_samples$sigma_eps[-(1:burn_in)])

# Trace plot
# par(mfrow = c(2,1))
# traceplot(mcmce.post_samples_phi, main = "Trace plot for MCMC with exact likelihood")
# traceplot(mcmcw.post_samples_phi, main = "Trace plot for MCMC with Whittle likelihood")

#########################
###        STAN       ###
#########################

hmc_filepath <- paste0(result_directory, "hmc_results_n", n, 
                       "_phi", phi_string, "_", date, ".rds")

# n_post_samples <- 10000
# burn_in <- 1000
stan.iters <- n_post_samples + burn_in

if (rerun_hmc) {
  stan_results <- run_hmc_lgss(data = y, iters = stan.iters, burn_in = burn_in)
  
  if (save_hmc_results) {
    saveRDS(stan_results, hmc_filepath)
  }
  
} else {
  stan_results <- readRDS(hmc_filepath)
}


# hmc.fit <- extract(hfit, pars = c("theta_phi", "theta_sigma"),
#                    permuted = F)
# 
# hmc.theta_phi <- hmc.fit[,,1]
# hmc.theta_sigma <- hmc.fit[,,2]

hmc.post_samples_phi <- stan_results$draws[,,1]#tanh(hmc.theta_phi)
hmc.post_samples_eta <- stan_results$draws[,,2]#sqrt(exp(hmc.theta_sigma))
hmc.post_samples_eps <- stan_results$draws[,,3]#sqrt(exp(hmc.theta_sigma))


################################################################################
##                            Posterior densities                             ##
################################################################################

par(mfrow = c(1,3))
# plot(density(mcmce.post_samples_phi), main = "Posterior of phi", col = "blue")
plot(density(hmc.post_samples_phi), main = "Posterior of phi", col = "blue")
lines(density(mcmcw.post_samples_phi), col = "blue", lty = 2)
lines(density(rvgaw.post_samples_phi), col = "red", lty = 2)
abline(v = phi, lty = 2)
legend("topright", legend = c("HMC", "MCMC Whittle", "R-VGA Whittle"), 
       col = c("blue", "blue", "red"), lty = c(1, 2, 2))

# plot(density(mcmce.post_samples_eta), main = "Posterior of sigma_eta", col = "blue")
plot(density(hmc.post_samples_eta), main = "Posterior of sigma_eta", col = "blue")
lines(density(mcmcw.post_samples_eta), col = "blue", lty = 2)
lines(density(rvgaw.post_samples_eta), col = "red", lty = 2)
abline(v = sigma_eta, lty = 2)
legend("topright", legend = c("HMC", "MCMC Whittle", "R-VGA Whittle"), 
       col = c("blue", "blue", "red"), lty = c(1, 2, 2))

# plot(density(mcmce.post_samples_eps), xlim = c(sigma_eps - 0.1, sigma_eps + 0.15),
#      main = "Posterior of sigma_epsilon", col = "blue")
plot(density(hmc.post_samples_eps), xlim = c(sigma_eps - 0.1, sigma_eps + 0.15),
     main = "Posterior of sigma_epsilon", col = "blue")
lines(density(mcmcw.post_samples_eps), col = "blue", lty = 2)
lines(density(rvgaw.post_samples_eps), col = "red", lty = 2)
abline(v = sigma_eps, lty = 2)
legend("topright", legend = c("HMC", "MCMC Whittle", "R-VGA Whittle"), 
       col = c("blue", "blue", "red"), lty = c(1, 2, 2))

# ## Trajectories
if (transform == "arctanh") {
  mu_phi <- sapply(rvgaw_results$mu, function(x) x[1])
  mu_sigma_eta <- sapply(rvgaw_results$mu, function(x) x[2])
  mu_sigma_eps <- sapply(rvgaw_results$mu, function(x) x[3])

  mu_phi <- tanh(mu_phi)
  mu_sigma_eta <- sqrt(exp(mu_sigma_eta))
  mu_sigma_eps <- sqrt(exp(mu_sigma_eps))

  # rvgae_mu <- tanh(unlist(rvgae_results$mu)[plot_range])
} else {
  # rvgae_mu <- unlist(rvgae_results$mu)[plot_range]
  # rvgae_mu <- exp(rvgae_mu) / (1 + exp(rvgae_mu))
  rvgaw_mu <- unlist(rvgaw_results$mu)
  rvgaw_mu <- exp(rvgaw_mu) / (1 + exp(rvgaw_mu))
}
plot_range <- 1:length(mu_phi) #400:1000#floor(n/2)
# if (reorder_freq) {
#   plot_title <- c("Trajectories with randomly reordered frequencies")
# } else {
#   plot_title <- c("Trajectories with original order of frequencies")
# }


par(mfrow = c(1,3))
plot(mu_phi[plot_range], type = "l",
     ylab = "phi", xlab = "Iterations", main = "Trajectory of phi")
abline(h = phi, lty = 2)

plot(mu_sigma_eta[plot_range], type = "l",
     ylab = "sigma_eta", xlab = "Iterations", main = "Trajectory of sigma_eta")
abline(h = sigma_eta, lty = 2)

plot(mu_sigma_eps[plot_range], type = "l",
     ylab = "sigma_eps", xlab = "Iterations", main = "Trajectory of sigma_eps")
abline(h = sigma_eps, lty = 2)
