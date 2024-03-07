## R-VGAW on the SV model

## MCMC on multivariate SV model
rm(list = ls())
setwd("~/R-VGA-Whittle/Multi_SV_Sigma/")

## Flags
# date <- "20230920" #"20230918" has 5D, "20230920" has 3D
date <- "20230918" #"20240128
regenerate_data <- F
save_data <- F
use_cholesky <- T # use lower Cholesky factor to parameterise Sigma_eta
transform <- "arctanh"
prior_type <- "prior1"
use_heaps_mapping <- F
plot_likelihood_surface <- F
plot_prior_samples <- F
plot_trace <- F
plot_trace_hmcw <- T
plot_trajectories <- F
save_plots <- F

rerun_rvgaw <- F
rerun_mcmcw <- F
rerun_hmc <- F
rerun_hmcw <- T

save_rvgaw_results <- F
save_mcmcw_results <- F
save_hmc_results <- F
save_hmcw_results <- F

## R-VGAW flags
use_tempering <- F
temper_first <- F
reorder_freq <- T
decreasing <- T
use_median <- F

library("mvtnorm")
library("astsa")
library("stcos")
library("coda")
library(Matrix)
# library("expm")
reticulate::use_condaenv("myenv", required = TRUE)
library(tensorflow)
tfp <- import("tensorflow_probability")
tfd <- tfp$distributions
library(keras)
library("cmdstanr")

source("./source/run_rvgaw_multi_sv.R")
source("./source/run_mcmc_multi_sv.R")
source("./source/compute_whittle_likelihood_multi_sv.R")
source("./source/run_mcmc_sv.R")
source("./source/compute_whittle_likelihood_sv.R")
source("./source/construct_prior2.R")
source("./source/map_functions.R")
source("./source/construct_Sigma.R")
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

#######################
##   Generate data   ##
#######################

dataset <- "5" #"hmc_est" 

Tfin <- 10000
d <- 2L

## Result directory
result_directory <- paste0("./results/", d, "d/")

if (regenerate_data) {
  
  # sigma_eta1 <- sqrt(0.2)#0.1
  # sigma_eta2 <- sqrt(0.1) #0.05
  # Sigma_eta <- diag(c(sigma_eta1^2, sigma_eta2^2))
  # 
  # sigma_eps1 <- 1 #0.01
  # sigma_eps2 <- 1 #0.02
  # 
  # Sigma_eps <- diag(c(sigma_eps1, sigma_eps2))
  
  Phi <- diag(0.1*c(9:(9-d+1)))
  Sigma_eps <- diag(d)
  
  if (dataset == "0") {
    Sigma_eta <- diag(0.1*(1:d))
    
    # } else if (dataset == "2") {
    #   Phi <- matrix(c(-0.7, 0.4, 0.9, 0.7), 2, 2, byrow = T)
    # } else if (dataset == "3") {
    #   phi11 <- 0.9 #0.9
    #   phi12 <- 0.3 #0.1 # dataset1: 0.1, dataset2 : -0.5
    #   phi21 <- 0#.4 #0.2 # dataset1: 0.2, dataset2 : -0.1
    #   phi22 <- 0.9 #0.7
    #   Phi <- matrix(c(phi11, phi12, phi21, phi22), 2, 2, byrow = T)
    # } else if (dataset == "4") {
    #   Phi <- matrix(c(-0.9, 0.8, -0.2, -0.4), 2, 2, byrow = T)
  } else if (dataset == "5") {
    # Phi <- diag(c(0.9, 0.8))
    nlower <- d*(d-1)/2
    diags <- 0.1*(1:d)
    lowers <- 0.05*(1:nlower)
    Sigma_eta <- diag(diags)
    Sigma_eta[lower.tri(Sigma_eta)] <- lowers
    Sigma_eta[upper.tri(Sigma_eta)] <- t(Sigma_eta)[upper.tri(Sigma_eta)]
    
  } else if (dataset == "hmc_est") {
    Phi <- diag(c(0.96, 0.97))
    Sigma_eta <- matrix(c(0.18, 0.11, 0.11, 0.125), 2, 2)
  } else { # generate a random Phi matrix
    Phi <- matrix(c(0.7, 0, 0, 0.8), 2, 2)
    Sigma_eta <- matrix(c(0.4, 0.05, 0.05, 0.2), 2, 2)
  } 
  
  x1 <- rep(0, d)
  X <- matrix(NA, nrow = Tfin+1, ncol = d) # x_0:T
  X[1, ] <- t(x1)
  Y <- matrix(NA, nrow = Tfin, ncol = d) # y_1:T
  # Y[, 1] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
  set.seed(2024)
  for (t in 1:Tfin) {
    X[t+1, ] <- Phi %*% X[t, ] + t(rmvnorm(1, rep(0, d), Sigma_eta))
    V <- diag(exp(X[t+1, ]/2))
    Y[t, ] <- V %*% t(rmvnorm(1, rep(0, d), Sigma_eps))
  }
  
  multi_sv_data <- list(X = X, Y = Y, Phi = Phi, Sigma_eta = Sigma_eta, 
                        Sigma_eps = Sigma_eps)
  
  if (save_data) {
    saveRDS(multi_sv_data, file = paste0("./data/multi_sv_data_", d, "d_Tfin", Tfin, "_", date, "_", dataset, ".rds"))
  }
} else {
  multi_sv_data <- readRDS(file = paste0("./data/multi_sv_data_", d, "d_Tfin", Tfin, "_", date, "_", dataset, ".rds"))
  
  X <- multi_sv_data$X
  Y <- multi_sv_data$Y
  Phi <- multi_sv_data$Phi
  Sigma_eta <- multi_sv_data$Sigma_eta
  Sigma_eps <- multi_sv_data$Sigma_eps
}

par(mfrow = c(ceiling(d/2),2))
# plot(X[1, ], type = "l")
# plot(X[2, ], type = "l")
# 
for (k in 1:d) {
  plot(Y[, k], type = "l")
}

############################## Inference #######################################

## Change prior to new set of parameters -- maybe just put priors on Phi_11, Phi_22
## Constrain so that Phi_11 and Phi_22 are both in (-1,1) -- use arctanh() for this?
## Parameterise Sigma_eta = LL^T and put prior on LL^T


## Construct initial distribution/prior
prior <- construct_prior(data = Y, prior_type = prior_type, use_cholesky = use_cholesky)
prior_mean <- prior$prior_mean
prior_var <- prior$prior_var

d <- as.integer(ncol(Y))
Tfin <- as.integer(nrow(Y))
param_dim <- length(prior_mean)


####?############################
##    R-VGAW implementation   ##
################################

if (use_tempering) {
  n_temper <- 50
  K <- 100
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

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_Tfin", Tfin, 
                         temper_info, reorder_info, "_", date, "_", dataset, "_", 
                         prior_type, ".rds")

if (rerun_rvgaw) {
  rvgaw_results <- run_rvgaw_multi_sv(data = Y, prior_mean = prior_mean, 
                                      prior_var = prior_var, S = S,
                                      use_cholesky = use_cholesky,
                                      transform = transform,
                                      use_tempering = use_tempering, 
                                      temper_first = temper_first,
                                      temper_schedule = temper_schedule, 
                                      reorder_freq = reorder_freq, 
                                      decreasing = decreasing,
                                      use_median = use_median)
  if (save_rvgaw_results) {
    saveRDS(rvgaw_results, rvgaw_filepath)
  }
  
} else {
  rvgaw_results <- readRDS(rvgaw_filepath)
}

rvgaw.post_samples_Phi <- rvgaw_results$post_samples$Phi
rvgaw.post_samples_Sigma_eta <- rvgaw_results$post_samples$Sigma_eta

# rvgaw.post_samples_phi_11 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[1,1]))
# # rvgaw.post_samples_phi_12 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[1,2]))
# # rvgaw.post_samples_phi_21 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[2,1]))
# rvgaw.post_samples_phi_22 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[2,2]))
# 
# rvgaw.post_samples_sigma_eta_11 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[1,1]))
# rvgaw.post_samples_sigma_eta_22 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[2,2]))
# 
# if (use_cholesky) {
#   rvgaw.post_samples_sigma_eta_12 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[1,2]))
#   rvgaw.post_samples_sigma_eta_21 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[2,1]))
# }

# par(mfrow = c(3,2))
# plot(density(rvgaw.post_samples_phi_11), col = "royalblue", main = "phi_11")
# abline(v = Phi[1,1], lty = 2)
# # plot(density(rvgaw.post_samples_phi_12), col = "royalblue", main = "phi_12")
# # abline(v = Phi[1,2], lty = 2)
# # plot(density(rvgaw.post_samples_phi_21), col = "royalblue", main = "phi_21")
# # abline(v = Phi[2,1], lty = 2)
# plot(density(rvgaw.post_samples_phi_22), col = "royalblue", main = "phi_22")
# abline(v = Phi[2,2], lty = 2)
# 
# plot(density(rvgaw.post_samples_sigma_eta_11), col = "royalblue", main = "sigma_eta_11")
# abline(v = Sigma_eta[1,1], lty = 2)
# 
# if (use_cholesky) {
#   plot(density(rvgaw.post_samples_sigma_eta_12), col = "royalblue", main = "sigma_eta_12")
#   abline(v = Sigma_eta[1,2], lty = 2)
#   plot(density(rvgaw.post_samples_sigma_eta_21), col = "royalblue", main = "sigma_eta_21")
#   abline(v = Sigma_eta[2,1], lty = 2)
# }
# 
# plot(density(rvgaw.post_samples_sigma_eta_22), col = "royalblue", main = "sigma_eta_22")
# abline(v = Sigma_eta[2,2], lty = 2)

#############################
##   MCMC implementation   ##
#############################
print("Starting MCMC with Whittle likelihood...")

mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_results_Tfin", Tfin, 
                         "_", date, "_", dataset, "_", prior_type, ".rds")

n_post_samples <- 10000
burn_in <- 5000
iters <- n_post_samples + burn_in

if (rerun_mcmcw) {
  mcmcw_results <- run_mcmc_multi_sv(data = Y, iters = iters, burn_in = burn_in, 
                                     prior_mean = prior_mean, prior_var = prior_var,
                                     adapt_proposal = T, use_whittle_likelihood = T,
                                     use_cholesky = use_cholesky, transform = transform)
  if (save_mcmcw_results) {
    saveRDS(mcmcw_results, mcmcw_filepath)
  }
} else {
  mcmcw_results <- readRDS(mcmcw_filepath)
}

## Extract samples
mcmcw.post_samples_Phi <- lapply(mcmcw_results$post_samples, function(x) x$Phi) #post_samples_theta[, 1]
mcmcw.post_samples_Sigma_eta <- lapply(mcmcw_results$post_samples, function(x) x$Sigma_eta) #post_samples_theta[, 2]
# 
# mcmcw.post_samples_phi_11 <- lapply(mcmcw.post_samples_Phi, function(x) x[1,1])
# mcmcw.post_samples_phi_12 <- lapply(mcmcw.post_samples_Phi, function(x) x[1,2])
# mcmcw.post_samples_phi_21 <- lapply(mcmcw.post_samples_Phi, function(x) x[2,1])
# mcmcw.post_samples_phi_22 <- lapply(mcmcw.post_samples_Phi, function(x) x[2,2])
# 
# mcmcw.post_samples_phi_11 <- as.mcmc(unlist(mcmcw.post_samples_phi_11[-(1:burn_in)]))
# mcmcw.post_samples_phi_12 <- as.mcmc(unlist(mcmcw.post_samples_phi_12[-(1:burn_in)]))
# mcmcw.post_samples_phi_21 <- as.mcmc(unlist(mcmcw.post_samples_phi_21[-(1:burn_in)]))
# mcmcw.post_samples_phi_22 <- as.mcmc(unlist(mcmcw.post_samples_phi_22[-(1:burn_in)]))
# 
# mcmcw.post_samples_sigma_eta_11 <- lapply(mcmcw.post_samples_Sigma_eta, function(x) x[1,1])
# mcmcw.post_samples_sigma_eta_12 <- lapply(mcmcw.post_samples_Sigma_eta, function(x) x[1,2])
# mcmcw.post_samples_sigma_eta_21 <- lapply(mcmcw.post_samples_Sigma_eta, function(x) x[2,1])
# mcmcw.post_samples_sigma_eta_22 <- lapply(mcmcw.post_samples_Sigma_eta, function(x) x[2,2])
# 
# mcmcw.post_samples_sigma_eta_11 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_11[-(1:burn_in)]))
# mcmcw.post_samples_sigma_eta_12 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_12[-(1:burn_in)]))
# mcmcw.post_samples_sigma_eta_21 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_21[-(1:burn_in)]))
# mcmcw.post_samples_sigma_eta_22 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_22[-(1:burn_in)]))


# par(mfrow = c(3,2))
# coda::traceplot(mcmcw.post_samples_phi_11, main = "Trace plot for phi_11")
# abline(h = Phi[1,1], col = "red", lty = 2)
# # coda::traceplot(mcmcw.post_samples_phi_12, main = "Trace plot for phi_12")
# # abline(h = Phi[1,2], col = "red", lty = 2)
# # coda::traceplot(mcmcw.post_samples_phi_21, main = "Trace plot for phi_21")
# # abline(h = Phi[2,1], col = "red", lty = 2)
# coda::traceplot(mcmcw.post_samples_phi_22, main = "Trace plot for phi_22")
# abline(h = Phi[2,2], col = "red", lty = 2)
# 
# coda::traceplot(mcmcw.post_samples_sigma_eta_11, main = "Trace plot for sigma_eta_11")
# abline(h = Sigma_eta[1,1], col = "red", lty = 2)
# 
# if (use_cholesky) {
#   coda::traceplot(mcmcw.post_samples_sigma_eta_12, main = "Trace plot for sigma_eta_12")
#   abline(h = Sigma_eta[1,2], col = "red", lty = 2)
#   coda::traceplot(mcmcw.post_samples_sigma_eta_21, main = "Trace plot for sigma_eta_21")
#   abline(h = Sigma_eta[2,1], col = "red", lty = 2)
# }
# coda::traceplot(mcmcw.post_samples_sigma_eta_22, main = "Trace plot for sigma_eta_22")
# abline(h = Sigma_eta[2,2], col = "red", lty = 2)

########################
###       STAN       ###
########################
print("Starting HMC...")

hmc_filepath <- paste0(result_directory, "hmc_results_Tfin", Tfin, 
                       "_", date, "_", dataset, "_", prior_type, ".rds")


if (rerun_hmc) {
  
  n_post_samples <- 10000
  burn_in <- 5000
  stan.iters <- n_post_samples + burn_in
  d <- as.integer(ncol(Y))
  
  use_chol <- 0
  if (use_cholesky) {
    use_chol <- 1
  }
  
  if (use_heaps_mapping) {
    stan_file <- "./source/stan_multi_sv_heaps.stan"
    multi_sv_data <- list(d = ncol(Y), p = 1, Tfin = nrow(Y), Y = Y,
                          prior_mean_A = prior_mean[c(1,3,2,4)], diag_prior_var_A = diag(prior_var)[c(1,3,2,4)],
                          prior_mean_gamma = prior_mean[5:7], diag_prior_var_gamma = diag(prior_var)[5:7]
    )
  } else {
    stan_file <- "./source/stan_multi_sv.stan"
    multi_sv_data <- list(d = d, Tfin = Tfin, Y = Y,
                          prior_mean_Phi = prior_mean[1:d], 
                          diag_prior_var_Phi = diag(prior_var)[1:d],
                          prior_mean_gamma = prior_mean[(d+1):param_dim], 
                          diag_prior_var_gamma = diag(prior_var)[(d+1):param_dim],
                          transform = ifelse(transform == "arctanh", 1, 0)
    )
    
  }
  
  multi_sv_model <- cmdstan_model(
    stan_file,
    cpp_options = list(stan_threads = TRUE)
  )
  
  fit_stan_multi_sv <- multi_sv_model$sample(
    multi_sv_data,
    chains = 4,
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

######################################
##   Stan with Whittle likelihood   ##
######################################

hmcw_filepath <- paste0(result_directory, "hmcw_results_Tfin", Tfin, 
                       "_", date, "_", dataset, "_", prior_type, ".rds")

if (rerun_hmcw) {
  print("Starting HMC with Whittle likelihood...")
  
  n_post_samples <- 1000
  burn_in <- 500
  stan.iters <- n_post_samples + burn_in
  
  stan_file_whittle <- "./source/stan_multi_sv_whittle.stan"
  
  # ## Calculation of Whittle likelihood
  ## Fourier frequencies
  k <- seq(-ceiling(Tfin/2)+1, floor(Tfin/2), 1)
  k_in_likelihood <- k [k >= 1 & k <= floor((Tfin-1)/2)]
  freq <- 2 * pi * k_in_likelihood / Tfin
  
  # ## astsa package
  Z <- log(Y^2) - colMeans(log(Y^2))
  fft_out <- mvspec(Z, detrend = F, plot = F)
  I <- fft_out$fxx
  
  I_indices <- seq(dim(I)[3])
  I_list <- lapply(I_indices[1:length(freq)], function(x) I[,,x])

  re_matrices <- lapply(1:length(freq), function(i) Re(I_list[[i]]))
  im_matrices <- lapply(1:length(freq), function(i) Im(I_list[[i]]))
    
  # periodogram_array <- array(NA, dim = c(length(freq), d, d))
  # for(q in 1:length(freq)) {
  #   periodogram_array[q,,] <- I_list[[q]]
  # }
  # periodogram_array <- I_list
  # transform <- "arctanh"
  multi_sv_data_whittle <- list(d = ncol(Y), nfreq = length(freq), freqs = freq,
                                # periodogram = periodogram_array,
                                re_matrices = re_matrices,
                                im_matrices = im_matrices,
                                prior_mean_Phi = prior_mean[1:d], 
                                diag_prior_var_Phi = diag(prior_var)[1:d],
                                prior_mean_gamma = prior_mean[(d+1):param_dim], 
                                diag_prior_var_gamma = diag(prior_var)[(d+1):param_dim],
                                # diag_prior_var_gamma = rep(0.1, 3),
                                transform = ifelse(transform == "arctanh", 1, 0),
                                truePhi = Phi,
                                trueSigma = Sigma_eta
                                )
  
  multi_sv_model_whittle <- cmdstan_model(
    stan_file_whittle,
    cpp_options = list(stan_threads = TRUE)
  )
  
  fit_stan_multi_sv_whittle <- multi_sv_model_whittle$sample(
    multi_sv_data_whittle,
    chains = 1,
    threads = parallel::detectCores(),
    refresh = 100,
    iter_warmup = burn_in,
    iter_sampling = n_post_samples
  )
  
  stan_whittle_results <- list(draws = fit_stan_multi_sv_whittle$draws(variables = c("Phi_mat", "Sigma_eta_mat")),
                       time = fit_stan_multi_sv_whittle$time)
  
  if (save_hmcw_results) {
    saveRDS(stan_whittle_results, hmcw_filepath)
  }
  
} else {
  stan_whittle_results <- readRDS(hmcw_filepath)
}

hmcw.post_samples_Phi <- stan_whittle_results$draws[,,1:4]
hmcw.post_samples_Sigma_eta <- stan_whittle_results$draws[,,5:8]


## Plot posterior estimates
indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
hmc_indices <- diag(matrix(1:d^2, d, d, byrow = T))
par(mfrow = c(d+1,d))

plot_margin <- c(-0.1, 0.1)

### Posterior of diagonal entries of Phi  
for (k in 1:d) {    
  rvgaw.post_samples_phi <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[k,k]))
  mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[k,k]))
  # hmc.post_samples_phi <- unlist(lapply(hmc.post_samples_Phi, function(x) x[k,k]))
  
  ind <- paste0(k,k)
  plot(density(rvgaw.post_samples_phi), col = "red", lty = 3, lwd = 2, 
       main = bquote(phi[.(ind)]), xlim = Phi[k,k] + plot_margin)
  lines(density(mcmcw.post_samples_phi), col = "royalblue", lty = 3, lwd = 2)
  # lines(density(hmc.post_samples_phi), col = "deepskyblue")
  
  lines(density(hmc.post_samples_Phi[,,hmc_indices[k]]), col = "deepskyblue", lty = 1, lwd = 2)
  lines(density(hmcw.post_samples_Phi[,,hmc_indices[k]]), col = "goldenrod", lty = 1, lwd = 2)
  
  abline(v = Phi[k,k], lty = 2, lwd = 2)
  # legend("topright", legend = c("R-VGAW", "HMC"), col = c("red", "forestgreen"),
  #        lty = c(2,2,1), cex = 0.7)
}

### Posterior of Sigma_eta
# par(mfrow = c(1,d))
hmc_indices <- c(matrix(1:(d^2), d, d)) #c(1,5,9)
# for (k in 1:d) {
for (k in 1:nrow(indices)) {
  i <- indices[k, 1]
  j <- indices[k, 2]
  
  rvgaw.post_samples_sigma_eta <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[i,j]))
  mcmcw.post_samples_sigma_eta <- unlist(lapply(mcmcw.post_samples_Sigma_eta, function(x) x[i,j]))
  
  ind <- paste0(i,j)
  plot(density(rvgaw.post_samples_sigma_eta), col = "red", lty = 3, lwd = 2, 
       main = bquote(sigma_eta[.(ind)]), xlim = Sigma_eta[i,j] + plot_margin)
  lines(density(mcmcw.post_samples_sigma_eta), col = "royalblue", lty = 3, lwd = 2)
  lines(density(hmc.post_samples_Sigma_eta[,,hmc_indices[k]]), lty = 1, col = "deepskyblue", lwd = 2)
  lines(density(hmcw.post_samples_Sigma_eta[,,hmc_indices[k]]), lty = 1, col = "goldenrod", lwd = 2)
  
  abline(v = Sigma_eta[i,j], lty = 2, lwd = 2)
  # legend("topright", legend = c("R-VGAW", "HMC"), col = c("red", "forestgreen"),
  #        lty = c(2,2,1), cex = 0.3, y.intersp = 0.25)
}

# par(mfrow = c(2,4))
# 
# plot_margin <- c(-0.2, 0.2)
# plot(density(mcmcw.post_samples_phi_11), col = "blue", lty = 2, lwd = 1.5, 
#      main = "phi_11", 
#      xlim = c(Phi[1,1] + plot_margin))
# lines(density(rvgaw.post_samples_phi_11), col = "red", lty = 2, lwd = 1.5,)
# lines(density(hmc.post_samples_Phi[,,1]), col = "forestgreen", lwd = 1.5,)
# lines(density(hmcw.post_samples_Phi[,,1]), col = "goldenrod", lwd = 1.5,)
# # lines(density(mcmcw1.post_samples_phi), col = "green")
# abline(v = Phi[1,1], lty = 2)
# legend("topright", legend = c("MCMCW", "R-VGAW", "HMC"), col = c("blue", "red", "forestgreen"),
#        lty = c(2,2,1), cex = 0.7)
# 
# plot(density(mcmcw.post_samples_phi_12), col = "blue", lty = 2, lwd = 1.5,
#      main = "phi_12", 
#      xlim = c(Phi[1,2] + plot_margin))
# lines(density(rvgaw.post_samples_phi_12), col = "red", lty = 2, lwd = 1.5,)
# lines(density(hmc.post_samples_Phi[,,3]), col = "forestgreen", lwd = 1.5,)
# lines(density(hmcw.post_samples_Phi[,,3]), col = "goldenrod", lwd = 1.5,)
# abline(v = Phi[1,2], lty = 2)
# 
# plot(density(mcmcw.post_samples_phi_21), col = "blue", lty = 2, lwd = 1.5, 
#      main = "phi_21", 
#      xlim = c(Phi[2,1] + plot_margin))
# lines(density(rvgaw.post_samples_phi_21), col = "red", lty = 2, lwd = 1.5)
# lines(density(hmc.post_samples_Phi[,,2]), col = "forestgreen", lwd = 1.5)
# lines(density(hmcw.post_samples_Phi[,,2]), col = "goldenrod", lwd = 1.5)
# abline(v = Phi[2,1], lty = 2)
# 
# plot(density(mcmcw.post_samples_phi_22), col = "blue", lty = 2, lwd = 1.5, 
#      main = "phi_22", 
#      xlim = c(Phi[2,2] + plot_margin))
# lines(density(rvgaw.post_samples_phi_22), col = "red", lty = 2, lwd = 1.5)
# lines(density(hmc.post_samples_Phi[,,4]), col = "forestgreen", lwd = 1.5)
# lines(density(hmcw.post_samples_Phi[,,4]), col = "goldenrod", lwd = 1.5)
# abline(v = Phi[2,2], lty = 2)
# 
# plot(density(mcmcw.post_samples_sigma_eta_11), col = "blue", lty = 2, lwd = 1.5,
#      main = "sigma_eta_11", xlim = c(Sigma_eta[1,1] + c(-0.2, 0.2)))
# lines(density(rvgaw.post_samples_sigma_eta_11), col = "red", lty = 2, lwd = 1.5)
# lines(density(hmc.post_samples_Sigma_eta[,,1]), col = "forestgreen", lwd = 1.5)
# lines(density(hmcw.post_samples_Sigma_eta[,,1]), col = "goldenrod", lwd = 1.5)
# # lines(density(mcmcw1.post_samples_sigma_eta2), col = "green")
# abline(v = Sigma_eta[1,1], lty = 2)
# 
# if (use_cholesky) {
#   plot(density(mcmcw.post_samples_sigma_eta_12), col = "blue", lty = 2, lwd = 1.5,
#        main = "sigma_eta_12", xlim = c(Sigma_eta[1,2] + c(-0.2, 0.2)))
#   lines(density(rvgaw.post_samples_sigma_eta_12), col = "red", lty = 2, lwd = 1.5)
#   abline(v = Sigma_eta[1,2], lty = 2)
#   
#   plot(density(mcmcw.post_samples_sigma_eta_21), col = "blue", lty = 2, lwd = 1.5,
#        main = "sigma_eta_21")
#   lines(density(rvgaw.post_samples_sigma_eta_21), col = "red", lty = 2, lwd = 1.5)
#   abline(v = Sigma_eta[2,1], lty = 2)
# }
# 
# plot(density(mcmcw.post_samples_sigma_eta_22), col = "blue", lty = 2, lwd = 1.5, 
#      main = "sigma_eta_22", xlim = c(Sigma_eta[2,2] + c(-0.2, 0.2)))
# lines(density(rvgaw.post_samples_sigma_eta_22), col = "red", lty = 2, lwd = 1.5)
# lines(density(hmc.post_samples_Sigma_eta[,,4]), col = "forestgreen", lwd = 1.5)
# lines(density(hmcw.post_samples_Sigma_eta[,,4]), col = "goldenrod", lwd = 1.5)
# abline(v = Sigma_eta[2,2], lty = 2)
# 
# 

## Plot trace
if (plot_trace_hmcw) {
  par(mfrow = c(4,2))
  for (k in 1:8) {
    if (k <= 4) {
      param_name = "phi"
      ind <- as.numeric(indices[k, ])
    } else {
      param_name = "sigma_eta"
      ind <- as.numeric(indices[k-4, ])
    }
    
    plot(stan_whittle_results$draws[,,k], type = "l", ylab = "param", 
         main = bquote(.(param_name)[.(ind)]))
  }
}


#CHECK COMPUTATIONS OF THE SPECTRAL DENSITY
## test likelihood computation
params_true <- list(Phi = Phi, Sigma_eta = Sigma_eta)

## 2. Calculate likelihood
out <- compute_whittle_likelihood_multi_sv(Y = Z, fourier_freqs = freq,
                                           periodogram = I[,,1:length(freq)],
                                           params = params_true,
                                           use_tensorflow = F)
ind <- 1
spec_dens <- out$spec_dens[[ind]]
# Phi_inv <- out$Phi_inv[[ind]]
llh <- out$log_likelihood_parts[ind]

## manually invert a 2x2 complex matrix
# periodogram <- out$periodogram[,,ind]
# mat <- spec_dens
# detmat <- mat[1,1] * mat[2,2] - mat[2,1] * mat[1,2] 
# invmat <- 1/detmat * matrix(c(mat[2,2], - mat[2,1], -mat[1,2], mat[1,1]), 2, 2)
# sum(diag(periodogram %*% invmat))
