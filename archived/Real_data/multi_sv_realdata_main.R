setwd("~/R-VGA-Whittle/Real_data")

rm(list = ls())
# library("coda")
library("mvtnorm")
library("cmdstanr")
library("astsa")
# library("stcos")
library("dplyr")
library(tensorflow)
# reticulate::use_condaenv("tf2.11", required = TRUE)
library(keras)

source("./source/run_rvgaw_multi_sv.R")
source("./source/run_mcmc_multi_sv.R")
source("./source/compute_whittle_likelihood_multi_sv.R")
# source("./source/run_mcmc_sv.R")
# source("./source/compute_whittle_likelihood_sv.R")
source("./source/construct_prior2.R")
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
date <- "20230823"
use_cholesky <- T
rerun_rvgaw <- F
rerun_mcmcw <- F
rerun_hmc <- F

save_rvgaw_results <- F
save_mcmcw_results <- F
save_hmc_results <- F

## R-VGAW flags
use_tempering <- T
reorder_freq <- T
decreasing <- T

#########################
##      Read data      ##
#########################

dataset <- "monthly" # daily or monthly

if (dataset == "daily") {
  returns_data <- read.csv("./data/5_Industry_Portfolios_Daily_cleaned.csv")
  datafile <- "_daily"
  Y <- returns_data[, 2:4]
} else if (dataset == "daily10000") {
  returns_data <- read.csv("./data/5_Industry_Portfolios_Daily_cleaned.csv")
  datafile <- "_daily10000"
  Y <- returns_data[1:10000, 2:3]
} else {
  returns_data <- read.csv("./data/5_Industry_Portfolios_cleaned.CSV")
  datafile <- ""
  Y <- returns_data[1:100, 2:3]
}

# Y <- returns_data[, 2:4]
Y_mean_corrected <- Y - colMeans(Y)
d <- ncol(Y_mean_corrected)
# param_dim <- d + d + d*(d-1)/2

par(mfrow = c(d, 1))
for (c in 1:ncol(Y_mean_corrected)) {
  plot(Y_mean_corrected[, c], type = "l")
}

## Construct initial distribution/prior
prior <- construct_prior(data = Y, use_cholesky = use_cholesky)
prior_mean <- prior$prior_mean
prior_var <- prior$prior_var

param_dim <- length(prior_mean)

browser()
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

S <- 100L
a_vals <- 1

################ R-VGA starts here #################
print("Starting R-VGAL with Whittle likelihood...")

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_svdata", datafile, 
                         temper_info, reorder_info, "_", date, ".rds")

if (rerun_rvgaw) {
  rvgaw_results <- run_rvgaw_multi_sv(data = Y_mean_corrected, prior_mean = prior_mean, 
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

#############################
##   MCMC implementation   ##
#############################
print("Starting MCMC with Whittle likelihood...")

mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_results_svdata", datafile, 
                         "_", date, ".rds")

n_post_samples <- 5000
burn_in <- 5000
iters <- n_post_samples + burn_in

if (rerun_mcmcw) {
  mcmcw_results <- run_mcmc_multi_sv(data = Y_mean_corrected, iters = iters, burn_in = burn_in, 
                                     prior_mean = prior_mean, prior_var = prior_var,
                                     adapt_proposal = F, use_whittle_likelihood = T)
  if (save_mcmcw_results) {
    saveRDS(mcmcw_results, mcmcw_filepath)
  }
} else {
  mcmcw_results <- readRDS(mcmcw_filepath)
}

## Extract samples
mcmcw.post_samples_Phi <- lapply(mcmcw_results$post_samples, function(x) x$Phi) #post_samples_theta[, 1]
mcmcw.post_samples_Sigma_eta <- lapply(mcmcw_results$post_samples, function(x) x$Sigma_eta) #post_samples_theta[, 2]

########################
###       STAN       ###
########################
print("Starting HMC...")

hmc_filepath <- paste0(result_directory, "hmc_results_svdata", datafile,  
                       "_", date, ".rds")

if (rerun_hmc) {
  
  # n_post_samples <- 10000
  # burn_in <- 1000
  # stan.iters <- n_post_samples + burn_in
  
  stan_file <- "./source/stan_multi_sv.stan"
  
  multi_sv_model <- cmdstan_model(
    stan_file,
    cpp_options = list(stan_threads = TRUE)
  )
  
  # hmc_indices <- c(matrix(1:(d^2), d, d, byrow = T)) # HMC vech() goes col-wise so need to modify the order of parameters a bit
  # multi_sv_data <- list(d = ncol(Y_mean_corrected), Tfin = nrow(Y_mean_corrected), 
  #                       Y = Y_mean_corrected,
  #                       prior_mean_A = prior_mean[hmc_indices], 
  #                       diag_prior_var_A = diag(prior_var)[hmc_indices],
  #                       prior_mean_gamma = prior_mean[(d^2+1):param_dim], 
  #                       diag_prior_var_gamma = diag(prior_var)[(d^2+1):param_dim]
  # )
  
  multi_sv_data <- list(d = ncol(Y_mean_corrected), Tfin = nrow(Y_mean_corrected), 
                        Y = Y_mean_corrected,
                        prior_mean_Phi = prior_mean[1:d], diag_prior_var_Phi = diag(prior_var)[1:d],
                        prior_mean_gamma = prior_mean[(d+1):param_dim], diag_prior_var_gamma = diag(prior_var)[(d+1):param_dim],
                        use_chol = 0)
  
  
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

hmc.post_samples_Phi <- stan_results$draws[,,1:d]
hmc.post_samples_Sigma_eta <- stan_results$draws[,,(d+1):(param_dim)]

## Plot posterior estimates
indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
hmc_indices <- c(matrix(1:d^2, d, d, byrow = T))
par(mfrow = c(d+1,d))

# param_names <- c("phi_11", "phi_12", "phi_21", "phi_22")
for (k in 1:nrow(indices)) {
  i <- indices[k, 1]
  j <- indices[k, 2]
  
  rvgaw.post_samples_phi <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[i,j]))
  mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[i,j]))
  
  plot(density(rvgaw.post_samples_phi), col = "red", lty = 2, 
       main = bquote(phi[.(c(i,j))]))
  lines(density(mcmcw.post_samples_phi[-(1:burn_in)]), col = "blue", lty = 2)
  lines(density(hmc.post_samples_Phi[,,hmc_indices[k]]), col = "forestgreen")
  # legend("topright", legend = c("R-VGAW", "HMC"), col = c("red", "forestgreen"),
  #        lty = c(2,2,1), cex = 0.7)
}

# par(mfrow = c(1,d))
hmc_indices <- diag(matrix(1:(d^2), d, d))
for (k in 1:d) {
  rvgaw.post_samples_sigma_eta <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[k,k]))
  mcmcw.post_samples_sigma_eta <- unlist(lapply(mcmcw.post_samples_Sigma_eta, function(x) x[k,k]))
  
  plot(density(rvgaw.post_samples_sigma_eta), col = "red", lty = 2,
       main = bquote(sigma_eta[.(c(k,k))]))
  lines(density(mcmcw.post_samples_sigma_eta[-(1:burn_in)]), col = "blue", lty = 2)
  lines(density(hmc.post_samples_Sigma_eta[,,hmc_indices[k]]), col = "forestgreen")
  # legend("topright", legend = c("R-VGAW", "HMC"), col = c("red", "forestgreen"),
  #        lty = c(2,2,1), cex = 0.3, y.intersp = 0.25)
}

## red = R-VGA Whittle, blue = MCMC Whittle, green = HMC
