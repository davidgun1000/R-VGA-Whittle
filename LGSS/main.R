setwd("~/R-VGA-whittle/LGSS/")
rm(list = ls())

# library("mvtnorm")
source("./source/run_rvgaw_lgss.R")
source("./source/run_mcmc_lgss.R")
source("./source/compute_kf_likelihood.R")
source("./source/compute_whittle_likelihood_lgss.R")
source("./source/update_sigma.R")

library(coda)
result_directory <- "./results/"

## Flags
date <- "20230525"
regenerate_data <- T
save_data <- T

rerun_rvgaw <- T
rerun_mcmce <- T
rerun_mcmcw <- T
save_rvgaw_results <- T
save_mcmce_results <- T
save_mcmcw_results <- T

## R-VGA flags
use_tempering <- T
reorder_freq <- F
transform <- "logit"

## MCMC flags
adapt_proposal <- T

## True parameters
sigma_eps <- 0.1 # measurement error var
sigma_eta <- 0.9 # process error var
phi <- 0.9

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

# Generate true process x_1:T
n <- 10000
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
burn_in <- 1000
n_post_samples <- 10000
MCMC_iters <- n_post_samples + burn_in # Number of MCMC iterations
accept <- rep(0, MCMC_iters)
acceptProb <- c()
mcmc.post_samples <- c()

## Prior
prior_mean <- 0
prior_var <- 1

## Initial state mean and variance for the KF
state_ini_mean <- 0
state_ini_var <- 1

# ## Test the likelihood computation by plotting likelihood surface over a grid of parameter values
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
#      xlab = "phi", ylab = "log likelihood", main = "Exact likelihood")
# legend("topleft", legend = c("true value", "arg max llh"),
#        col = c("black", "red"), lty = 2, cex = 0.5)
# abline(v = phi_grid[which.max(likelihood_exact)], lty = 1, col = "red")
# abline(v = phi, lty = 2)
# 
# plot(phi_grid[plot_range], likelihood_whittle[plot_range], type = "l",
#      xlab = "phi", ylab = "log likelihood", main = "Whittle likelihood")
# legend("topleft", legend = c("true value", "arg max llh"),
#        col = c("black", "red"), lty = 2, cex = 0.5)
# abline(v = phi_grid[which.max(likelihood_whittle)], lty = 1, col = "red")
# abline(v = phi, lty = 2)

################################################################################
##                      R-VGA with Whittle likelihood                         ##
################################################################################

if (use_tempering) {
  n_temper <- 0.1 * n
  temper_schedule <- rep(1/10, 10)
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


if (rerun_rvgaw) {
  
  
  rvgaw_results <- run_rvgaw_lgss(y = y, sigma_eta = sigma_eta, sigma_eps = sigma_eps, 
                                  prior_mean = prior_mean, prior_var = prior_var, 
                                  S = 500, use_tempering = use_tempering, 
                                  n_temper = n_temper,
                                  temper_schedule = temper_schedule,
                                  transform = transform)
  if (save_rvgaw_results) {
    saveRDS(rvgaw_results, rvgaw_filepath)
  }
  
} else {
  rvgaw_results <- readRDS(rvgaw_filepath)
}

rvgaw.post_samples <- rvgaw_results$post_samples

################################################################################
##                        MCMC with exact likelihood                          ##   
################################################################################

mcmce_filepath <- paste0(result_directory, "mcmc_exact_results_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

if (rerun_mcmce) {
  mcmce_results <- run_mcmc_lgss(y, sigma_eta, sigma_eps, iters = MCMC_iters, burn_in = burn_in,
                                 prior_mean = prior_mean, prior_var = prior_var, 
                                 state_ini_mean = state_ini_mean, state_ini_var = state_ini_var,
                                 adapt_proposal = T, use_whittle_likelihood = F)
  
  if (save_mcmce_results) {
    saveRDS(mcmce_results, mcmce_filepath)
  }
  
} else {
  mcmce_results <- readRDS(mcmce_filepath)
}

mcmce.post_samples <- as.mcmc(mcmce_results$post_samples[-(1:burn_in)])

################################################################################
##                       MCMC with Whittle likelihood                         ##
################################################################################

mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_results_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

if (rerun_mcmcw) {
  mcmcw_results <- run_mcmc_lgss(y, sigma_eta, sigma_eps, iters = MCMC_iters, burn_in = burn_in,
                                 prior_mean = prior_mean, prior_var = prior_var,  
                                 state_ini_mean = state_ini_mean, state_ini_var = state_ini_var,
                                 adapt_proposal = T, use_whittle_likelihood = T)
  
  if (save_mcmcw_results) {
    saveRDS(mcmcw_results, mcmcw_filepath)
  }
} else {
  mcmcw_results <- readRDS(mcmcw_filepath)
}

mcmcw.post_samples <- as.mcmc(mcmcw_results$post_samples[-(1:burn_in)])

# Trace plot
par(mfrow = c(2,1))
traceplot(mcmce.post_samples, main = "Trace plot for MCMC with exact likelihood")
traceplot(mcmcw.post_samples, main = "Trace plot for MCMC with Whittle likelihood")

################################################################################
##                            Posterior densities                             ##
################################################################################

par(mfrow = c(1,1))
plot(density(mcmce.post_samples), main = "Posterior of phi", col = "blue")
lines(density(mcmcw.post_samples), col = "blue", lty = 2)
lines(density(rvgaw.post_samples), col = "red", lty = 2)
abline(v = phi, lty = 2)
legend("topright", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"), 
       col = c("blue", "blue", "red"), lty = c(1, 2, 2))

## Trajectories
if (transform == "arctanh") {
  # rvgae_mu <- tanh(unlist(rvgae_results$mu)[plot_range])
  rvgaw_mu <- tanh(unlist(rvgaw_results$mu))
} else {
  # rvgae_mu <- unlist(rvgae_results$mu)[plot_range]
  # rvgae_mu <- exp(rvgae_mu) / (1 + exp(rvgae_mu))
  rvgaw_mu <- unlist(rvgaw_results$mu)
  rvgaw_mu <- exp(rvgaw_mu) / (1 + exp(rvgaw_mu))
}
plot_range <- 1:length(rvgaw_mu) #400:1000#floor(n/2)
if (reorder_freq) {
  plot_title <- c("Trajectories with randomly reordered frequencies")
} else {
  plot_title <- c("Trajectories with original order of frequencies")
}

plot(rvgaw_mu[plot_range], type = "l",
     ylab = "mu", xlab = "Iterations", main = plot_title)
abline(h = phi, lty = 2)
# legend("bottomright", legend = c("R-VGA Exact", "R-VGA Whittle"), col = c("black", "red"),
#        lty = 1)
