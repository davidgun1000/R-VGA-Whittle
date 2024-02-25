## R-VGAW on the SV model

## MCMC on multivariate SV model
rm(list = ls())
setwd("~/R-VGA-Whittle/Multi_SV_real/")

library("coda")
library("mvtnorm")
# library("astsa")
library("cmdstanr")
# library("expm")
# library("stcos")
library(tensorflow)
library(stringr)
reticulate::use_condaenv("tf2.11", required = TRUE)
library(keras)
library(Matrix)

source("./source/compute_whittle_likelihood_sv.R")
source("./source/run_rvgaw_sv.R")
source("./source/run_mcmc_sv.R")
source("./source/run_hmc_sv.R")

# source("./source/run_corr_pmmh_sv.R")
# source("./source/particleFilter.R")

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

date <- "20230918" #"20230626" # the 20230626 version has sigma_eta = 0.7, the 20230918 version has sigma_eta = sqrt(0.1)
# date <- "20230626"

## R-VGA flags
# regenerate_data <- F
# save_data <- F
use_tempering <- T
reorder_freq <- F
decreasing <- T
reorder_seed <- 2024
plot_likelihood_surface <- F
plot_trajectories <- T
prior_type <- "prior1"
transform <- "logit"

## Flags
rerun_rvgaw <- F
rerun_mcmcw <- F
# rerun_mcmce <- F
rerun_hmc <- F

save_rvgaw_results <- F
save_mcmcw_results <- F
# save_mcmce_results <- F
save_hmc_results <- F
save_plots <- F

#########################
##      Read data      ##
#########################

dataset <- "49_industries" #49_industries, developed_6
datafreq <- "daily" # daily or monthly
column <- "Books"
nstocks <- 1
nobs <- "all" 

Y <- c()
if (str_detect(datafreq, "daily")) {
  
  if (dataset == "49_industries") {
    returns_data <- read.csv("./data/49_Industry_Portfolios_Daily.csv")
    if (nobs == "all") {
      nobs <- nrow(returns_data)
    }
    # Y <- returns_data$Toys[1:nobs]
    Y <- returns_data[[column]][1:nobs]
    
  } else if (dataset == "developed_6") {
    returns_data <- read.csv("./data/Developed_6_Portfolios_ME_OP_Daily.csv")
    
    if (nobs == "all") {
      nobs <- nrow(returns_data)
    }
    Y <- returns_data$SMALL.LoBO[1:nobs]
    
  } else {
    returns_data <- read.csv("./data/5_Industry_Portfolios_Daily_cleaned.csv")
    
    if (nobs == "all") {
      nobs <- nrow(returns_data)
    }  
    Y <- returns_data[1:nobs, 2:(2+nstocks-1)]
    
  }
  datafile <- paste0("_daily", nobs)
  
} else { # monthly
  returns_data <- read.csv("./data/5_Industry_Portfolios_cleaned.CSV")
  datafile <- paste0("_monthly", nobs)
  Y <- returns_data[1:nobs, 2:(2+nstocks-1)]
}

# Y <- returns_data[, 2:4]
Y_mean_corrected <- Y - mean(Y)
d <- ncol(Y_mean_corrected)
# par(mfrow = c(2,1))
plot(Y, type = "l")

## Test likelihood computation
if (plot_likelihood_surface) {
  param_grid <- seq(0.01, 0.99, length.out = 100)
  llh1 <- c() # log likelihood when varying phi
  llh2 <- c() # log likelihood when varying sigma_eta
  
  for (k in 1:length(param_grid)) {
    params1 <- list(phi = param_grid[k], sigma_eta = sigma_eta, sigma_xi = sqrt(pi^2/2))
    params2 <- list(phi = phi, sigma_eta = param_grid[k], sigma_xi = sqrt(pi^2/2))
    
    llh1[k] <- compute_whittle_likelihood_sv(y = y, params = params1)
    llh2[k] <- compute_whittle_likelihood_sv(y = y, params = params2)
    
  }
  
  par(mfrow = c(1,2))
  plot_range <- 1:100
  plot(param_grid[plot_range], llh1[plot_range], type = "l", 
       xlab = "Parameter", ylab = "Log likelihood", main = "phi")
  abline(v = phi, lty = 2)
  abline(v = param_grid[which.max(llh1)], col = "red", lty = 2)
  
  plot_range <- 1:100
  plot(param_grid[plot_range], llh2[plot_range], type = "l", 
       xlab = "Parameter", ylab = "Log likelihood", main = "sigma_eta")
  abline(v = sigma_eta, lty = 2)
  abline(v = param_grid[which.max(llh2)], col = "red", lty = 2)
}
# 
# test_eps <- rnorm(10000, 0, 1)
# test_xi <- log(test_eps^2)
# var(test_xi)

# # Test exact likelihood
# phi_grid <- seq(0.01, 0.99, length.out = 100)
# llh <- c()
# 
# for (k in 1:length(phi_grid)) {
#   params_pf <- list(phi = phi_grid[k], sigma_eta = sigma_eta, sigma_eps = sigma_eps,
#                     sigma_xi = sqrt(pi^2/2))
#   pf_out <- particleFilter(y = y, N = 100, iniState = 0, param = params_pf)
#   llh[k] <- pf_out$log_likelihood
# }
# 
# plot(phi_grid, llh, type = "l")
# abline(v = phi_grid[which.max(llh)], col = "red", lty = 2)
# 

## Setting up result directory

result_directory <- paste0("./results/", dataset, "/", datafreq, "_", 
                           nstocks, "stocks/", transform, "/")

if (dataset == "49_industries") {
  column <- paste0("_", column)
} else{
  column <- ""
}

########################################
##                R-VGA               ##
########################################

S <- 100

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

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_realdata", 
                         column, "_T", nobs,
                         temper_info, reorder_info, "_", date, ".rds")

## Prior
if (prior_type == "prior1") {
  prior_mean <- c(0, -1) #rep(0,2)
  prior_var <- diag(c(1, 0.1)) #diag(1, 2)
} else {
  prior_mean <- c(0, -0) #rep(0,2)
  prior_var <- diag(c(1, 1)) #diag(1, 2)
}

prior_theta <- rmvnorm(10000, prior_mean, prior_var)
prior_phi <- tanh(prior_theta[, 1])
prior_eta <- sqrt(exp(prior_theta[, 2]))
# prior_xi <- sqrt(exp(prior_theta[, 3]))
# par(mfrow = c(2,1))
# hist(prior_phi)
# hist(prior_eta)

if (rerun_rvgaw) {
  rvgaw_results <- run_rvgaw_sv(y = Y_mean_corrected, #sigma_eta = sigma_eta, sigma_eps = sigma_eps, 
                                prior_mean = prior_mean, prior_var = prior_var, 
                                deriv = "tf", 
                                S = S, use_tempering = use_tempering, 
                                reorder_freq = reorder_freq,
                                decreasing = decreasing, 
                                n_temper = n_temper,
                                temper_schedule = temper_schedule,
                                transform = transform)
  
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

mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_results_realdata",
                         column, "_T", nobs,
                         "_", date, ".rds")

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
  mcmcw_results <- run_mcmc_sv(Y_mean_corrected, #sigma_eta, sigma_eps, 
                               iters = MCMC_iters, burn_in = burn_in,
                               prior_mean = prior_mean, prior_var = prior_var,  
                               state_ini_mean = state_ini_mean, state_ini_var = state_ini_var,
                               adapt_proposal = T, use_whittle_likelihood = T,
                               transform = transform)
  
  if (save_mcmcw_results) {
    saveRDS(mcmcw_results, mcmcw_filepath)
  }
} else {
  mcmcw_results <- readRDS(mcmcw_filepath)
}

mcmcw.post_samples_phi <- as.mcmc(mcmcw_results$post_samples$phi[-(1:burn_in)])
mcmcw.post_samples_eta <- as.mcmc(mcmcw_results$post_samples$sigma_eta[-(1:burn_in)])
# mcmcw.post_samples_xi <- as.mcmc(mcmcw_results$post_samples$sigma_xi[-(1:burn_in)])

# par(mfrow = c(2,1))
# coda::traceplot(mcmcw.post_samples_phi, main = "Trace plot for phi")
# coda::traceplot(mcmcw.post_samples_eta, main = "Trace plot for sigma_eta")
# # traceplot(mcmcw.post_samples_xi, main = "Trace plot for sigma_xi")
# 
# par(mfrow = c(1,2))
# plot(density(mcmcw.post_samples_phi), main = "Posterior of phi", 
#      col = "blue", lty = 2, lwd = 2)
# lines(density(rvgaw.post_samples_phi), col = "red", lty = 2, lwd = 2)
# abline(v = phi, lty = 3)
# legend("topleft", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
#        col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)
# 
# plot(density(mcmcw.post_samples_eta), main = "Posterior of sigma_eta", 
#      col = "blue", lty = 2, lwd = 2)
# lines(density(rvgaw.post_samples_eta), col = "red", lty = 2, lwd = 2)
# abline(v = sigma_eta, lty = 3)
# legend("topright", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
#        col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)

# ####### MCMCE ##########
# mcmce_filepath <- paste0(result_directory, "mcmc_exact_results_n", n, 
#                          "_phi", phi_string, "_", date, ".rds")
# 
# if (rerun_mcmce) {
#   mcmce_results <- run_mcmc_sv(y, #sigma_eta, sigma_eps, 
#                                iters = MCMC_iters, burn_in = burn_in,
#                                prior_mean = prior_mean, prior_var = prior_var,  
#                                state_ini_mean = state_ini_mean, state_ini_var = state_ini_var,
#                                adapt_proposal = T, use_whittle_likelihood = F)
#   
#   if (save_mcmce_results) {
#     saveRDS(mcmce_results, mcmce_filepath)
#   }
# } else {
#   mcmce_results <- readRDS(mcmce_filepath)
# }
# 
# mcmce.post_samples_phi <- as.mcmc(mcmce_results$post_samples$phi[-(1:burn_in)])
# mcmce.post_samples_eta <- as.mcmc(mcmce_results$post_samples$sigma_eta[-(1:burn_in)])
# # mcmce.post_samples_xi <- as.mcmc(mcmce_results$post_samples$sigma_xi[-(1:burn_in)])
# 
# par(mfrow = c(2,1))
# coda::traceplot(mcmce.post_samples_phi, main = "Trace plot for phi")
# coda::traceplot(mcmce.post_samples_eta, main = "Trace plot for sigma_eta")
# # traceplot(mcmce.post_samples_xi, main = "Trace plot for sigma_xi")

### STAN ###
hmc_filepath <- paste0(result_directory, "hmc_results_realdata",
                       column, "_T", nobs,
                       "_", date, ".rds")

# n_post_samples <- 10000
# burn_in <- 1000
stan.iters <- n_post_samples + burn_in

if (rerun_hmc) {
  stan_results <- run_hmc_sv(data = Y_mean_corrected, 
                             transform = transform, 
                             iters = stan.iters, burn_in = burn_in)
  
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

###########################

par(mfrow = c(1,2))
plot(density(mcmcw.post_samples_phi), main = "Posterior of phi", xlim = c(0.9, 0.99), 
     col = "royalblue", lty = 2, lwd = 2)
# lines(density(mcmce.post_samples_phi), col = "blue", lwd = 2)
lines(density(rvgaw.post_samples_phi), col = "red", lty = 2, lwd = 2)
lines(density(hmc.post_samples_phi), col = "skyblue", lty = 1, lwd = 2)
legend("topleft", legend = c("HMC", "MCMC Whittle", "R-VGA Whittle"),
       col = c("skyblue", "royalblue", "red"), lty = c(1, 2, 2), cex = 0.7)

plot(density(mcmcw.post_samples_eta), main = "Posterior of sigma_eta", xlim = c(0.1, 1),
     col = "royalblue", lty = 2, lwd = 2)
# lines(density(mcmce.post_samples_eta), col = "blue", lwd = 2)
lines(density(rvgaw.post_samples_eta), col = "red", lty = 2, lwd = 2)
lines(density(hmc.post_samples_eta), col = "skyblue", lty = 1, lwd = 2)
legend("topright", legend = c("HMC", "MCMC Whittle", "R-VGA Whittle"),
       col = c("skyblue", "royalblue", "red"), lty = c(1, 2, 2), cex = 0.7)


## Trajectories
# mu_phi <- sapply(rvgaw_results$mu, function(x) x[1])
# mu_eta <- sapply(rvgaw_results$mu, function(x) x[2])
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
log_kappa2 <- mean(log(Y_mean_corrected^2)) - mean_log_eps2
kappa <- sqrt(exp(log_kappa2))

## Trajectories
if (plot_trajectories) {
  mu_phi <- sapply(rvgaw_results$mu, function(x) x[1])
  mu_eta <- sapply(rvgaw_results$mu, function(x) x[2])
  
  par(mfrow = c(1, 2))
  if (transform == "arctanh") {
    plot(tanh(mu_phi), type = "l", main = "Trajectory of phi")
  } else {
    plot(1 / (1 + exp(-mu_phi)), type = "l", main = "Trajectory of phi")
  }
  
  plot(sqrt(exp(mu_eta)), type = "l", main = "Trajectory of sigma_eta")
}


if (save_plots) {
  plot_file <- paste0("sv_posterior_1d", temper_info, reorder_info,
                      "_", date, ".png")
  filepath = paste0("./plots/", plot_file)
  png(filepath, width = 600, height = 350)
  
  par(mfrow = c(1,2))
  plot(density(mcmcw.post_samples_phi), main = "Posterior of phi", 
       col = "royalblue", lty = 2, lwd = 2)
  # lines(density(mcmce.post_samples_phi), col = "blue", lwd = 2)
  lines(density(rvgaw.post_samples_phi), col = "red", lty = 2, lwd = 2)
  lines(density(hmc.post_samples_phi), col = "skyblue", lty = 1, lwd = 2)
  abline(v = phi, lty = 3)
  legend("topleft", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
         col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)
  
  plot(density(mcmcw.post_samples_eta), main = "Posterior of sigma_eta", 
       col = "royalblue", lty = 2, lwd = 2)
  # lines(density(mcmce.post_samples_eta), col = "blue", lwd = 2)
  lines(density(rvgaw.post_samples_eta), col = "red", lty = 2, lwd = 2)
  lines(density(hmc.post_samples_eta), col = "skyblue", lty = 1, lwd = 2)
  abline(v = sigma_eta, lty = 3)
  legend("topright", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
         col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)
  
  dev.off()
}
