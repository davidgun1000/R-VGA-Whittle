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
regenerate_data <- F
save_data <- F
use_cholesky <- F # use lower Cholesky factor to parameterise Sigma_eta

rerun_rvgaw <- F
rerun_mcmcw <- F
rerun_hmc <- F

save_rvgaw_results <- F
save_mcmcw_results <- F
save_hmc_results <- F

## R-VGAW flags
use_tempering <- T
reorder_freq <- F
decreasing <- T

#######################
##   Generate data   ##
#######################

dataset <- "1"

Tfin <- 1000
if (regenerate_data) {
  phi11 <- 0.9
  phi12 <- 0.1 # dataset1: 0.1, dataset2 : -0.5
  phi21 <- 0.2 # dataset1: 0.2, dataset2 : -0.1
  phi22 <- 0.7
  
  if (dataset == "0") {
    Phi <- diag(c(phi11, phi22))  
  } else {
    Phi <- matrix(c(phi11, phi12, phi21, phi22), 2, 2, byrow = T)
  }
  
  sigma_eta1 <- 0.9#4
  sigma_eta2 <- 1.5 #1.5
  Sigma_eta <- diag(c(sigma_eta1, sigma_eta2))
  
  sigma_eps1 <- 1 #0.01
  sigma_eps2 <- 1 #0.02
  
  Sigma_eps <- diag(c(sigma_eps1, sigma_eps2))
  
  x1 <- c(0, 0)
  X <- matrix(NA, nrow = length(x1), ncol = Tfin+1) # x_0:T
  X[, 1] <- x1
  Y <- matrix(NA, nrow = length(x1), ncol = Tfin) # y_1:T
  # Y[, 1] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
  set.seed(2022)
  for (t in 1:Tfin) {
    X[, t+1] <- Phi %*% X[, t] + t(rmvnorm(1, c(0, 0), Sigma_eta))
    V <- diag(exp(X[, t+1]/2))
    Y[, t] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
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

par(mfrow = c(2,1))
plot(X[1, ], type = "l")
plot(X[2, ], type = "l")

plot(Y[1, ], type = "l")
plot(Y[2, ], type = "l")

############################## Inference #######################################

## Construct initial distribution/prior
prior <- construct_prior(data = Y)
prior_mean <- prior$prior_mean
prior_var <- prior$prior_var

param_dim <- length(prior_mean)

# ## Test run MCMC on a single series here ##
# n_post_samples <- 10000
# burn_in <- 5000
# MCMC_iters <- n_post_samples + burn_in
# 
# mcmcw1_results <- run_mcmc_sv(y = Y[1,], #sigma_eta, sigma_eps,
#                              iters = MCMC_iters, burn_in = burn_in,
#                              prior_mean = rep(0,2), prior_var = diag(c(1, 1)),
#                              state_ini_mean = state_ini_mean, state_ini_var = state_ini_var,
#                              adapt_proposal = T, use_whittle_likelihood = T)
# 
# mcmcw1.post_samples_phi <- as.mcmc(mcmcw1_results$post_samples$phi[-(1:burn_in)])
# mcmcw1.post_samples_sigma_eta2 <- as.mcmc(mcmcw1_results$post_samples$sigma_eta[-(1:burn_in)]^2)
# mcmcw.post_samples_xi <- as.mcmc(mcmcw_results$post_samples$sigma_xi[-(1:burn_in)])

################################
##    R-VGAW implementation   ##
################################

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

S <- 200L
a_vals <- 1

################ R-VGA starts here #################
print("Starting R-VGAL with Whittle likelihood...")

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_Tfin", Tfin, 
                         temper_info, reorder_info, "_", date, "_", dataset, ".rds")

if (rerun_rvgaw) {
  rvgaw_results <- run_rvgaw_multi_sv(data = Y, prior_mean = prior_mean, 
                                      prior_var = prior_var, S = S,
                                      use_tempering = use_tempering, 
                                      temper_schedule = temper_schedule, 
                                      reorder_freq = reorder_freq, decreasing = decreasing)
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

if (use_cholesky) {
  rvgaw.post_samples_sigma_eta_12 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[1,2]))
  rvgaw.post_samples_sigma_eta_21 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[2,1]))
}

par(mfrow = c(2,2))
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

if (use_cholesky) {
  plot(density(rvgaw.post_samples_sigma_eta_12), col = "blue", main = "sigma_eta_12")
  abline(v = Sigma_eta[1,2], lty = 2)
  plot(density(rvgaw.post_samples_sigma_eta_21), col = "blue", main = "sigma_eta_21")
  abline(v = Sigma_eta[2,1], lty = 2)
}

plot(density(rvgaw.post_samples_sigma_eta_22), col = "blue", main = "sigma_eta_22")
abline(v = Sigma_eta[2,2], lty = 2)

#############################
##   MCMC implementation   ##
#############################
print("Starting MCMC with Whittle likelihood...")

mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_results_Tfin", Tfin, 
                         "_", date, "_", dataset, ".rds")

n_post_samples <- 10000
burn_in <- 5000
iters <- n_post_samples + burn_in

if (rerun_mcmcw) {
  mcmcw_results <- run_mcmc_multi_sv(data = Y, iters = iters, burn_in = burn_in, 
                                     prior_mean = prior_mean, prior_var = prior_var,
                                     adapt_proposal = T, use_whittle_likelihood = T)
  if (save_mcmcw_results) {
    saveRDS(mcmcw_results, mcmcw_filepath)
  }
} else {
  mcmcw_results <- readRDS(mcmcw_filepath)
}

## Extract samples
mcmcw.post_samples_phi <- lapply(mcmcw_results$post_samples, function(x) x$Phi) #post_samples_theta[, 1]
mcmcw.post_samples_sigma_eta <- lapply(mcmcw_results$post_samples, function(x) x$Sigma_eta) #post_samples_theta[, 2]

mcmcw.post_samples_phi_11 <- lapply(mcmcw.post_samples_phi, function(x) x[1,1])
mcmcw.post_samples_phi_12 <- lapply(mcmcw.post_samples_phi, function(x) x[1,2])
mcmcw.post_samples_phi_21 <- lapply(mcmcw.post_samples_phi, function(x) x[2,1])
mcmcw.post_samples_phi_22 <- lapply(mcmcw.post_samples_phi, function(x) x[2,2])

mcmcw.post_samples_phi_11 <- as.mcmc(unlist(mcmcw.post_samples_phi_11[-(1:burn_in)]))
mcmcw.post_samples_phi_12 <- as.mcmc(unlist(mcmcw.post_samples_phi_12[-(1:burn_in)]))
mcmcw.post_samples_phi_21 <- as.mcmc(unlist(mcmcw.post_samples_phi_21[-(1:burn_in)]))
mcmcw.post_samples_phi_22 <- as.mcmc(unlist(mcmcw.post_samples_phi_22[-(1:burn_in)]))

mcmcw.post_samples_sigma_eta_11 <- lapply(mcmcw.post_samples_sigma_eta, function(x) x[1,1])
mcmcw.post_samples_sigma_eta_12 <- lapply(mcmcw.post_samples_sigma_eta, function(x) x[1,2])
mcmcw.post_samples_sigma_eta_21 <- lapply(mcmcw.post_samples_sigma_eta, function(x) x[2,1])
mcmcw.post_samples_sigma_eta_22 <- lapply(mcmcw.post_samples_sigma_eta, function(x) x[2,2])

mcmcw.post_samples_sigma_eta_11 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_11[-(1:burn_in)]))
mcmcw.post_samples_sigma_eta_12 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_12[-(1:burn_in)]))
mcmcw.post_samples_sigma_eta_21 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_21[-(1:burn_in)]))
mcmcw.post_samples_sigma_eta_22 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_22[-(1:burn_in)]))


par(mfrow = c(3,2))
coda::traceplot(mcmcw.post_samples_phi_11, main = "Trace plot for phi_11")
abline(h = Phi[1,1], col = "red", lty = 2)
coda::traceplot(mcmcw.post_samples_phi_12, main = "Trace plot for phi_12")
abline(h = Phi[1,2], col = "red", lty = 2)
coda::traceplot(mcmcw.post_samples_phi_21, main = "Trace plot for phi_21")
abline(h = Phi[2,1], col = "red", lty = 2)
coda::traceplot(mcmcw.post_samples_phi_22, main = "Trace plot for phi_22")
abline(h = Phi[2,2], col = "red", lty = 2)

coda::traceplot(mcmcw.post_samples_sigma_eta_11, main = "Trace plot for sigma_eta_11")
abline(h = Sigma_eta[1,1], col = "red", lty = 2)

if (use_cholesky) {
  coda::traceplot(mcmcw.post_samples_sigma_eta_12, main = "Trace plot for sigma_eta_12")
  abline(h = Sigma_eta[1,2], col = "red", lty = 2)
  coda::traceplot(mcmcw.post_samples_sigma_eta_21, main = "Trace plot for sigma_eta_21")
  abline(h = Sigma_eta[2,1], col = "red", lty = 2)
}
coda::traceplot(mcmcw.post_samples_sigma_eta_22, main = "Trace plot for sigma_eta_22")
abline(h = Sigma_eta[2,2], col = "red", lty = 2)

########################
###       STAN       ###
########################
print("Starting HMC...")

hmc_filepath <- paste0(result_directory, "hmc_results_Tfin", Tfin, 
                       "_", date, "_", dataset, ".rds")


if (rerun_hmc) {
  
  n_post_samples <- 10000
  burn_in <- 1000
  stan.iters <- n_post_samples + burn_in
  
  stan_file <- "./source/stan_multi_sv.stan"
  
  multi_sv_model <- cmdstan_model(
    stan_file,
    cpp_options = list(stan_threads = TRUE)
  )
  
  # multi_sv_data <- list(d = nrow(Y), Tfin = ncol(Y), Y = Y,
  #                       prior_mean_A = prior_mean[1:4], prior_var_A = prior_var[1:4, 1:4],
  #                       prior_mean_gamma = prior_mean[5:6], prior_var_gamma = prior_var[5:6, 5:6]
  # )
  
  multi_sv_data <- list(d = nrow(Y), Tfin = ncol(Y), Y = Y,
                        prior_mean_A = prior_mean[1:4], diag_prior_var_A = diag(prior_var)[1:4],
                        prior_mean_gamma = prior_mean[5:6], diag_prior_var_gamma = diag(prior_var)[5:6]
  )
  
  
  fit_stan_multi_sv <- multi_sv_model$sample(
    multi_sv_data,
    chains = 1,
    threads = parallel::detectCores(),
    refresh = 5,
    iter_warmup = burn_in,
    iter_sampling = n_post_samples
  )
  
  stan_results <- list(draws = fit_stan_multi_sv$draws(variables = c("Phi_mat", "Sigma_eta_mat")),
                       time = fit_stan_multi_sv$time)
  
  if (save_hmc_results) {
    saveRDS(stan_results, hmc_filepath)
  }
  
} else {
  stan_results <- readRDS(hmc_filepath)
}

hmc.post_samples_Phi <- stan_results$draws[,,1:4]
hmc.post_samples_Sigma_eta <- stan_results$draws[,,5:8]

## Posterior density comparisons

par(mfrow = c(2,4))
plot(density(mcmcw.post_samples_phi_11), col = "blue", lty = 2, main = "phi_11", 
     xlim = c(Phi[1,1] + c(-0.1, 0.1)))
lines(density(rvgaw.post_samples_phi_11), col = "red", lty = 2)
lines(density(hmc.post_samples_Phi[,,1]), col = "forestgreen")
# lines(density(mcmcw1.post_samples_phi), col = "green")
abline(v = Phi[1,1], lty = 2)

plot(density(mcmcw.post_samples_phi_12), col = "blue", lty = 2, main = "phi_12", 
     xlim = c(Phi[1,2] + c(-0.1, 0.1)))
lines(density(rvgaw.post_samples_phi_12), col = "red", lty = 2)
lines(density(hmc.post_samples_Phi[,,3]), col = "forestgreen")
abline(v = Phi[1,2], lty = 2)

plot(density(mcmcw.post_samples_phi_21), col = "blue", lty = 2, main = "phi_21", 
     xlim = c(Phi[2,1] + c(-0.1, 0.1)))
lines(density(rvgaw.post_samples_phi_21), col = "red", lty = 2)
lines(density(hmc.post_samples_Phi[,,2]), col = "forestgreen")
abline(v = Phi[2,1], lty = 2)

plot(density(mcmcw.post_samples_phi_22), col = "blue", lty = 2, main = "phi_22", 
     xlim = c(Phi[2,2] + c(-0.1, 0.1)))
lines(density(rvgaw.post_samples_phi_22), col = "red", lty = 2)
lines(density(hmc.post_samples_Phi[,,4]), col = "forestgreen")
abline(v = Phi[2,2], lty = 2)

plot(density(mcmcw.post_samples_sigma_eta_11), col = "blue", lty = 2, 
     main = "sigma_eta_11", xlim = c(Sigma_eta[1,1] + c(-0.25, 0.25)))
lines(density(rvgaw.post_samples_sigma_eta_11), col = "red", lty = 2)
lines(density(hmc.post_samples_Sigma_eta[,,1]), col = "forestgreen")
# lines(density(mcmcw1.post_samples_sigma_eta2), col = "green")
abline(v = Sigma_eta[1,1], lty = 2)

if (use_cholesky) {
  plot(density(mcmcw.post_samples_sigma_eta_12), col = "blue", lty = 2, 
       main = "sigma_eta_12", xlim = c(Sigma_eta[1,2] + c(-0.25, 0.25)))
  lines(density(rvgaw.post_samples_sigma_eta_12), col = "red", lty = 2)
  abline(v = Sigma_eta[1,2], lty = 2)

  plot(density(mcmcw.post_samples_sigma_eta_21), col = "blue", lty = 2, 
       main = "sigma_eta_21")
  lines(density(rvgaw.post_samples_sigma_eta_21), col = "red", lty = 2)
  abline(v = Sigma_eta[2,1], lty = 2)
}

plot(density(mcmcw.post_samples_sigma_eta_22), col = "blue", lty = 2, 
     main = "sigma_eta_22", xlim = c(Sigma_eta[2,2] + c(-0.25, 0.25)))
lines(density(rvgaw.post_samples_sigma_eta_22), col = "red", lty = 2)
lines(density(hmc.post_samples_Sigma_eta[,,4]), col = "forestgreen")
abline(v = Sigma_eta[2,2], lty = 2)


