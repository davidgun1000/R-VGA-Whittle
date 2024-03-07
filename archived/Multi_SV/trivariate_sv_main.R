## Trivariate SV model

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
prior_type <- "normal"

rerun_rvgaw <- F
rerun_mcmcw <- F
rerun_hmc <- F

save_rvgaw_results <- F
save_mcmcw_results <- F
save_hmc_results <- F

## R-VGA flags 
use_tempering <- T
reorder_freq <- T
decreasing <- T

## Plot options
plot_trace <- F

## MCMC options
n_post_samples <- 5000
burn_in <- 5000
iters <- n_post_samples + burn_in
#######################
##   Generate data   ##
#######################

dataset <- "2"

Tfin <- 1000
d <- 3

if (regenerate_data) {
  
  sigma_eta1 <- 0.9#4
  sigma_eta2 <- 1.5 #1.5
  sigma_eta3 <- 1.2 #1.5
  
  Sigma_eta <- diag(c(sigma_eta1, sigma_eta2, sigma_eta3))
  Sigma_eps <- diag(d)
  
  if (dataset == "0") {
    # Phi <- diag(c(0.7, 0.8, 0.9))
    Phi <- diag(c(0.9, 0.2, -0.8))
  } else if (dataset == "1") {
    Phi <- matrix(c(0.700, -0.025, 0.050,
                    -0.075, 0.800, 0.075,
                    -0.050, 0.025, 0.900), 3, 3, byrow = T)
  } else if (dataset == "2") {
    Phi <- matrix(c(-0.3, -0.1,  0.4,
                    -0.2,  0.9,  0.8,
                    -0.2, -0.7, -0.8), 3, 3, byrow = T)
  } else if (dataset == "3") {
    Phi <- matrix(c(0.9, 0.0, -0.1,
                    -0.6, 0.2, 0.5,
                    1.0, 0.3, -0.8), 3, 3, byrow = T)
    # diag(Phi) <- c(0.7, 0.8, 0.9)
  } else { # generate a random Phi matrix using the mapping from unconstrained to constrained VAR(1) coef 
    A <- matrix(rnorm(d^2), d, d)
    Phi <- backward_map(A, Sigma_eta)
    Phi <- round(Phi, digits = 1)
  }
  
  set.seed(2022)
  x1 <- rmvnorm(1, rep(0, d), compute_autocov_VAR1(Phi, Sigma_eta))
  X <- matrix(NA, nrow = length(x1), ncol = Tfin+1) # x_0:T
  X[, 1] <- x1
  Y <- matrix(NA, nrow = length(x1), ncol = Tfin) # y_1:T
  # Y[, 1] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
  for (t in 1:Tfin) {
    X[, t+1] <- Phi %*% X[, t] + t(rmvnorm(1, rep(0, d), Sigma_eta))
    V <- diag(exp(X[, t+1]/2))
    Y[, t] <- V %*% t(rmvnorm(1, rep(0, d), Sigma_eps))
  }
  
  multi_sv_data <- list(X = X, Y = Y, Phi = Phi, Sigma_eta = Sigma_eta, 
                        Sigma_eps = Sigma_eps)
  
  if (save_data) {
    saveRDS(multi_sv_data, file = paste0("./data/trivariate_sv_data_Tfin", Tfin, "_", date, "_", dataset, ".rds"))
  }
} else {
  multi_sv_data <- readRDS(file = paste0("./data/trivariate_sv_data_Tfin", Tfin, "_", date, "_", dataset, ".rds"))
  
  X <- multi_sv_data$X
  Y <- multi_sv_data$Y
  Phi <- multi_sv_data$Phi
  Sigma_eta <- multi_sv_data$Sigma_eta
  Sigma_eps <- multi_sv_data$Sigma_eps
}

par(mfrow = c(d,1))
plot(X[1, ], type = "l")
plot(X[2, ], type = "l")
plot(X[3, ], type = "l")

plot(Y[1, ], type = "l")
plot(Y[2, ], type = "l")
plot(Y[3, ], type = "l")

############################## Inference #######################################

## Construct initial distribution/prior
prior <- construct_prior(data = Y, prior_type = prior_type)
prior_mean <- prior$prior_mean
prior_var <- prior$prior_var

param_dim <- length(prior_mean)

## Prior
if (prior_type == "minnesota") {
  prior_type <- ""
} else {
  prior_type <- paste0("_", prior_type)
}


################################
##    R-VGAW implementation   ##
################################

if (use_tempering) {
  n_temper <- 10
  K <- 50
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

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_trivariate_Tfin", Tfin, 
                         temper_info, reorder_info, "_", date, "_", dataset,
                         prior_type, ".rds")

if (rerun_rvgaw) {
  rvgaw_results <- run_rvgaw_multi_sv(data = Y, prior_mean = prior_mean, 
                                      prior_var = prior_var, 
                                      n_post_samples = n_post_samples, S = S,
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

# rvgaw.post_samples_phi_11 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[1,1]))
# rvgaw.post_samples_phi_12 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[1,2]))
# rvgaw.post_samples_phi_21 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[2,1]))
# rvgaw.post_samples_phi_22 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[2,2]))
# 
# rvgaw.post_samples_sigma_eta_11 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[1,1]))
# rvgaw.post_samples_sigma_eta_22 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[2,2]))
# 
# if (use_cholesky) {
#   rvgaw.post_samples_sigma_eta_12 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[1,2]))
#   rvgaw.post_samples_sigma_eta_21 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[2,1]))
# }

#############################
##   MCMC implementation   ##
#############################
print("Starting MCMC with Whittle likelihood...")

mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_results_trivariate_Tfin", Tfin, 
                         "_", date, "_", dataset, prior_type, ".rds")

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
mcmcw.post_samples_Phi <- lapply(mcmcw_results$post_samples, function(x) x$Phi) #post_samples_theta[, 1]
mcmcw.post_samples_Sigma_eta <- lapply(mcmcw_results$post_samples, function(x) x$Sigma_eta) #post_samples_theta[, 2]

########################
###       STAN       ###
########################
print("Starting HMC...")

hmc_filepath <- paste0(result_directory, "hmc_results_trivariate_Tfin", Tfin, 
                       "_", date, "_", dataset, prior_type, ".rds")

if (rerun_hmc) {
  
  # n_post_samples <- 10000
  # burn_in <- 1000
  # stan.iters <- n_post_samples + burn_in
  
  stan_file <- "./source/stan_multi_sv.stan"
  
  multi_sv_model <- cmdstan_model(
    stan_file,
    cpp_options = list(stan_threads = TRUE)
  )
  
  # multi_sv_data <- list(d = nrow(Y), Tfin = ncol(Y), Y = Y,
  #                       prior_mean_A = prior_mean[1:4], prior_var_A = prior_var[1:4, 1:4],
  #                       prior_mean_gamma = prior_mean[5:6], prior_var_gamma = prior_var[5:6, 5:6]
  # )
  order <- c(matrix(1:(d^2), d, d, byrow = T))
  multi_sv_data <- list(d = nrow(Y), Tfin = ncol(Y), Y = Y,
                        prior_mean_A = prior_mean[order], diag_prior_var_A = diag(prior_var)[order],
                        prior_mean_gamma = prior_mean[(d^2+1):param_dim], 
                        diag_prior_var_gamma = diag(prior_var)[(d^2+1):param_dim]
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

hmc.post_samples_Phi <- stan_results$draws[,,1:(d^2)]
hmc.post_samples_Sigma_eta <- stan_results$draws[,,(d^2+1):(2*d^2)]

## Plot posterior estimates
indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
hmc_indices <- c(matrix(1:d^2, d, d, byrow = T))
par(mfrow = c(d+1,d))

for (k in 1:nrow(indices)) {
  i <- indices[k, 1]
  j <- indices[k, 2]
  
  rvgaw.post_samples_phi <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[i,j]))
  mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[i,j]))
  
  plot(density(rvgaw.post_samples_phi), col = "red", lty = 2, 
       main = bquote(phi[.(c(i,j))]), xlim = Phi[i,j] + c(-0.3, 0.3))
  lines(density(mcmcw.post_samples_phi), col = "blue", lty = 2)
  lines(density(hmc.post_samples_Phi[,,hmc_indices[k]]), col = "forestgreen")
  abline(v = Phi[i,j], lty = 2)
  # legend("topright", legend = c("R-VGAW", "HMC"), col = c("red", "forestgreen"),
  #        lty = c(2,2,1), cex = 0.7)
}

# par(mfrow = c(1,d))
hmc_indices <- diag(matrix(1:(d^2), d, d)) #c(1,5,9)
for (k in 1:d) {
  rvgaw.post_samples_sigma_eta <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[k,k]))
  mcmcw.post_samples_sigma_eta <- unlist(lapply(mcmcw.post_samples_Sigma_eta, function(x) x[k,k]))
  
  plot(density(rvgaw.post_samples_sigma_eta), col = "red", lty = 2,
       main = bquote(sigma_eta[.(c(k,k))]), xlim = Sigma_eta[k,k] + c(-0.4, 0.4))
  lines(density(mcmcw.post_samples_sigma_eta), col = "blue", lty = 2)
  lines(density(hmc.post_samples_Sigma_eta[,,hmc_indices[k]]), col = "forestgreen")
  abline(v = Sigma_eta[k,k], lty = 2)
  # legend("topright", legend = c("R-VGAW", "HMC"), col = c("red", "forestgreen"),
  #        lty = c(2,2,1), cex = 0.3, y.intersp = 0.25)
}

## Trace plots
if (plot_trace) {
  for (k in 1:nrow(indices)) {
    i <- indices[k, 1]
    j <- indices[k, 2]
    
    mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[i,j]))
    ## red = R-VGA Whittle, blue = MCMC Whittle, green = HMC
    coda::traceplot(coda::as.mcmc(mcmcw.post_samples_phi))
  }
  
  for (k in 1:d) {
    mcmcw.post_samples_sigma_eta <- unlist(lapply(mcmcw.post_samples_Sigma_eta, function(x) x[k,k]))
    coda::traceplot(coda::as.mcmc(mcmcw.post_samples_sigma_eta))
  }
  
}
