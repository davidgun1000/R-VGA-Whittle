setwd("~/R-VGA-Whittle/LGSS/")
rm(list = ls())

library(mvtnorm)
library(coda)
# library(Deriv)
# library(tensorflow)
# reticulate::use_condaenv("tf2.11", required = TRUE)
# library(keras)

# source("./source/run_rvgaw_lgss_tf.R")
source("./source/run_mcmc_lgss_allparams.R")
source("./source/compute_kf_likelihood.R")
source("./source/compute_whittle_likelihood_lgss.R")
source("./source/update_sigma.R")

## Result directory
result_directory <- "./results/"

## Flags
date <- "20230525"
regenerate_data <- T
save_data <- F

rerun_mcmce <- T
rerun_mcmcw <- T
save_mcmce_results <- F
save_mcmcw_results <- F

adapt_proposal <- T

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
  print("Generating data...")
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
sigma_eps_grid <- seq(0.05, 1, length.out = 100)
likelihood_whittle <- c()
likelihood_exact <- c()

for (i in 1:length(sigma_eps_grid)) {
  params_list <- list(phi = phi, sigma_eta = sigma_eta, sigma_eps = sigma_eps_grid[i])
  
  # if (use_whittle_likelihood) {
  likelihood_whittle[i] <- compute_whittle_likelihood_lgss(y = y, params = params_list)
  # } else {
  kf_out <- compute_kf_likelihood(state_prior_mean = state_ini_mean,
                                  state_prior_var = state_ini_var,
                                  iters = length(y), observations = y,
                                  params = params_list)

  likelihood_exact[i] <- kf_out$log_likelihood
  # }
}
par(mfrow = c(2,1))
# margin <- 5
plot_range <- 1:length(sigma_eps_grid)
# plot_range <- (which.max(likelihood_exact) - margin):(which.max(likelihood_exact) + 2*margin)
plot(sigma_eps_grid[plot_range], likelihood_exact[plot_range], type = "l", 
     xlab = "sigma_epsilon", ylab = "log likelihood", main = paste0("Exact likelihood (n = ", n, ")"))
legend("topleft", legend = c("true value", "arg max llh"),
       col = c("black", "red"), lty = 2, cex = 0.5)
abline(v = sigma_eps_grid[which.max(likelihood_exact)], lty = 2, col = "red")
abline(v = sigma_eps, lty = 2)

# plot_range <- (which.max(likelihood_whittle) - margin):(which.max(likelihood_whittle) + 2*margin)
plot(sigma_eps_grid[plot_range], likelihood_whittle[plot_range], type = "l",
     xlab = "phi", ylab = "log likelihood", main = paste0("Whittle likelihood (n = ", n, ")"))
abline(v = sigma_eps_grid[which.max(likelihood_whittle)], lty = 2, col = "red")
abline(v = sigma_eps, lty = 2)
legend("topleft", legend = c("true value", "arg max llh"),
       col = c("black", "red"), lty = 2, cex = 0.5)

browser()

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
##                        MCMC with exact likelihood                          ##   
################################################################################

mcmce_filepath <- paste0(result_directory, "mcmc_exact_results_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

if (rerun_mcmce) {
  mcmce_results <- run_mcmc_lgss(y, #sigma_eta, sigma_eps, 
                                 iters = MCMC_iters, burn_in = burn_in,
                                 prior_mean = prior_mean, prior_var = prior_var, 
                                 state_ini_mean = state_ini_mean, state_ini_var = state_ini_var,
                                 adapt_proposal = T, use_whittle_likelihood = F)
  
  if (save_mcmce_results) {
    saveRDS(mcmce_results, mcmce_filepath)
  }
  
} else {
  mcmce_results <- readRDS(mcmce_filepath)
}

mcmce.post_samples_phi <- as.mcmc(mcmce_results$post_samples$phi[-(1:burn_in)])
mcmce.post_samples_eta <- as.mcmc(mcmce_results$post_samples$sigma_eta[-(1:burn_in)])
mcmce.post_samples_eps <- as.mcmc(mcmce_results$post_samples$sigma_eps[-(1:burn_in)])

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
par(mfrow = c(2,1))
traceplot(mcmce.post_samples_phi, main = "Trace plot for MCMC with exact likelihood")
traceplot(mcmcw.post_samples_phi, main = "Trace plot for MCMC with Whittle likelihood")

################################################################################
##                            Posterior densities                             ##
################################################################################

par(mfrow = c(1,3))
plot(density(mcmce.post_samples_phi), main = "Posterior of phi", col = "blue")
lines(density(mcmcw.post_samples_phi), col = "blue", lty = 2)
abline(v = phi, lty = 2)
legend("topright", legend = c("MCMC exact", "MCMC Whittle"), # "R-VGA Whittle"), 
       col = c("blue", "blue"), lty = c(1, 2))

plot(density(mcmce.post_samples_eta), main = "Posterior of sigma_eta", col = "blue")
lines(density(mcmcw.post_samples_eta), col = "blue", lty = 2)
abline(v = sigma_eta, lty = 2)
legend("topright", legend = c("MCMC exact", "MCMC Whittle"), #"R-VGA Whittle"), 
       col = c("blue", "blue"), lty = c(1, 2))

plot(density(mcmce.post_samples_eps), xlim = c(sigma_eps - 0.1, sigma_eps + 0.1),
     main = "Posterior of sigma_epsilon", col = "blue")
lines(density(mcmcw.post_samples_eps), col = "blue", lty = 2)
abline(v = sigma_eps, lty = 2)
legend("topright", legend = c("MCMC exact", "MCMC Whittle"),#"R-VGA Whittle"), 
       col = c("blue", "blue"), lty = c(1, 2))

