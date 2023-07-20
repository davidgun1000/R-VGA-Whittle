## Stochastic volatility model
setwd("~/R-VGA-Whittle/SV/")

library(mvtnorm)
library(coda)
library(Deriv)
library(tensorflow)
reticulate::use_condaenv("tf2.11", required = TRUE)
library(keras)

source("./source/compute_whittle_likelihood_sv.R")
source("./source/run_rvgaw_sv_tf.R")
source("./source/run_mcmc_sv.R")
source("./source/particleFilter.R")

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
regenerate_data <- T
save_data <- F
use_tempering <- T
reorder_freq <- T
decreasing <- T
reorder_seed <- 2024

## Flags
rerun_rvgaw <- T
rerun_mcmcw <- T
rerun_mcmce <- F

save_rvgaw_results <- T
save_mcmcw_results <- T
save_mcmce_results <- F

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

########################################
##                MCMC                ## 
########################################

mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_results_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

adapt_proposal <- T

n_post_samples <- 10000
burn_in <- 5000
MCMC_iters <- n_post_samples + burn_in

# prior_mean <- rep(0, 3)
# prior_var <- diag(c(1, 1, 0.01))

# prior_samples <- rmvnorm(10000, prior_mean, prior_var)
# prior_samples_phi <- tanh(prior_samples[, 1])                                       
# hist(prior_samples_sigma_xi)

if (rerun_mcmcw) {
  mcmcw_results <- run_mcmc_sv(y, #sigma_eta, sigma_eps, 
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
# mcmcw.post_samples_xi <- as.mcmc(mcmcw_results$post_samples$sigma_xi[-(1:burn_in)])

par(mfrow = c(2,1))
traceplot(mcmcw.post_samples_phi, main = "Trace plot for phi")
traceplot(mcmcw.post_samples_eta, main = "Trace plot for sigma_eta")
# traceplot(mcmcw.post_samples_xi, main = "Trace plot for sigma_xi")

par(mfrow = c(1,2))
plot(density(mcmcw.post_samples_phi), main = "Posterior of phi", 
     col = "blue", lty = 2, lwd = 1.5)
lines(density(rvgaw.post_samples_phi), col = "red", lty = 2, lwd = 1.5)
abline(v = phi, lty = 3)
legend("topleft", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
       col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)

plot(density(mcmcw.post_samples_eta), main = "Posterior of sigma_eta", 
     col = "blue", lty = 2, lwd = 1.5)
lines(density(rvgaw.post_samples_eta), col = "red", lty = 2, lwd = 1.5)
abline(v = sigma_eta, lty = 3)
legend("topright", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
       col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)

# browser()

####### MCMCE ##########
mcmce_filepath <- paste0(result_directory, "mcmc_exact_results_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

if (rerun_mcmce) {
  mcmce_results <- run_mcmc_sv(y, #sigma_eta, sigma_eps, 
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
# mcmce.post_samples_xi <- as.mcmc(mcmce_results$post_samples$sigma_xi[-(1:burn_in)])

par(mfrow = c(2,1))
traceplot(mcmce.post_samples_phi, main = "Trace plot for phi")
traceplot(mcmce.post_samples_eta, main = "Trace plot for sigma_eta")
# traceplot(mcmce.post_samples_xi, main = "Trace plot for sigma_xi")

###########################

par(mfrow = c(1,2))
plot(density(mcmcw.post_samples_phi), main = "Posterior of phi", 
     col = "blue", lty = 2, lwd = 1.5)
lines(density(mcmce.post_samples_phi), col = "blue", lwd = 1.5)
lines(density(rvgaw.post_samples_phi), col = "red", lty = 2, lwd = 1.5)
abline(v = phi, lty = 3)
legend("topleft", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
       col = c("blue", "red"), lty = c(1, 2, 2), cex = 0.7)

plot(density(mcmcw.post_samples_eta), main = "Posterior of sigma_eta", 
     col = "blue", lty = 2, lwd = 1.5)
lines(density(mcmce.post_samples_eta), col = "blue", lwd = 1.5)
lines(density(rvgaw.post_samples_eta), col = "red", lty = 2, lwd = 1.5)
abline(v = sigma_eta, lty = 3)
legend("topright", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
       col = c("blue", "red"), lty = c(1, 2, 2), cex = 0.7)

# plot(density(mcmcw.post_samples_xi), 
#      main = "Posterior of sigma_xi", col = "blue", lty = 2, lwd = 1.5)
# lines(density(mcmce.post_samples_xi), col = "blue", lwd = 1.5)
# lines(density(rvgaw.post_samples_xi), col = "red", lty = 2, lwd = 1.5)
# 
# abline(v = sqrt(pi^2/2), lty = 3)
# legend("topright", legend = c("MCMC Whittle", "R-VGA Whittle"),
#        col = c("blue", "red"), lty = c(2, 2))


## Trajectories
# mu_phi <- sapply(rvgaw_results$mu, function(x) x[1])
# mu_eta <- sapply(rvgaw_results$mu, function(x) x[2])
# mu_xi <- sapply(rvgaw_results$mu, function(x) x[3])
# 
# par(mfrow = c(1, 3))
# plot(tanh(mu_phi), type = "l", main = "Trajectory of phi")
# abline(h = phi, lty = 2)
# 
# plot(sqrt(exp(mu_eta)), type = "l", main = "Trajectory of sigma_eta")
# abline(h = sigma_eta, lty = 2)
# 
# plot(sqrt(exp(mu_xi)), type = "l", main = "Trajectory of sigma_xi")
# abline(h = sqrt(pi^2/2), lty = 2)

## Estimation of kappa
mean_log_eps2 <- digamma(1/2) + log(2)
log_kappa2 <- mean(log(y^2)) - mean_log_eps2
kappa <- sqrt(exp(log_kappa2))
# Construct covariance matrix of x
# t_vec <- seq(0, n-1, 1)
# Sigma_x <- sigma_eta^2/(1-phi^2) * phi^abs(outer(t_vec, t_vec, "-")) ## need to replace these with estimates of sigma_eta and phi later
# 
# sigma_xi <- sqrt(pi^2/2)
# Sigma_xi <- diag(sigma_xi^2, n)
# Sigma <- Sigma_x + Sigma_xi
# Sigma_inv <- chol2inv(chol(Sigma))
# z <- log(y^2)
# x_tilde <- (diag(n) - sigma_xi^2 * Sigma_inv) %*% z +
#             sigma_xi^2 * Sigma_inv %*% rep(1, n) #* mean(u) # this is z_tilde in Nguyen (2022)
# y_tilde <- y * exp(-x_tilde/2)
# kappa_tilde <- sqrt(mean(y_tilde))
