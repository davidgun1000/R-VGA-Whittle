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
library(Matrix)

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
prior_type <- "minnesota"
use_heaps_mapping <- F
plot_likelihood_surface <- F

rerun_rvgaw <- F
rerun_mcmcw <- F
rerun_hmc <- F
rerun_hmcw <- T

save_rvgaw_results <- F
save_mcmcw_results <- F
save_hmc_results <- F
save_hmcw_results <- F


## R-VGAW flags
use_tempering <- T
reorder_freq <- T
decreasing <- T

#######################
##   Generate data   ##
#######################

dataset <- "3" 

Tfin <- 1000
if (regenerate_data) {
  
  sigma_eta1 <- 0.6#0.1
  sigma_eta2 <- 0.3 #0.05
  Sigma_eta <- diag(c(sigma_eta1, sigma_eta2))
  
  sigma_eps1 <- 1 #0.01
  sigma_eps2 <- 1 #0.02
  
  Sigma_eps <- diag(c(sigma_eps1, sigma_eps2))
  
  if (dataset == "0") {
    Phi <- diag(c(0.9, 0.7))  
  } else if (dataset == "2") {
    Phi <- matrix(c(-0.7, 0.4, 0.9, 0.7), 2, 2, byrow = T)
  } else if (dataset == "3") {
    phi11 <- 0.5 #0.9
    phi12 <- 0.3 #0.1 # dataset1: 0.1, dataset2 : -0.5
    phi21 <- 0.4 #0.2 # dataset1: 0.2, dataset2 : -0.1
    phi22 <- 0.6 #0.7
    Phi <- matrix(c(phi11, phi12, phi21, phi22), 2, 2, byrow = T)
  } else if (dataset == "4") {
    Phi <- matrix(c(-0.9, 0.8, -0.2, -0.4), 2, 2, byrow = T)
  } else if (dataset == "5") {
    phi11 <- 0.5 #0.9
    phi12 <- 0.3 #0.1 # dataset1: 0.1, dataset2 : -0.5
    phi21 <- 0.4 #0.2 # dataset1: 0.2, dataset2 : -0.1
    phi22 <- 0.6 #0.7
    Phi <- matrix(c(phi11, phi12, phi21, phi22), 2, 2, byrow = T)
    
    sigma_eta1 <- 0.1
    sigma_eta2 <- 0.05
    Sigma_eta <- diag(c(sigma_eta1, sigma_eta2))
    
  }
  
  else { # generate a random Phi matrix
    d <- 2
    A <- matrix(rnorm(d^2), d, d)
    Phi <- backward_map(A, Sigma_eta)
    Phi <- round(Phi, digits = 1)
    print(Phi)
  }
  
  x1 <- c(0, 0)
  X <- matrix(NA, nrow = length(x1), ncol = Tfin+1) # x_0:T
  X[, 1] <- x1
  Y <- matrix(NA, nrow = length(x1), ncol = Tfin) # y_1:T
  # Y[, 1] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
  set.seed(2023)
  for (t in 1:Tfin) {
    X[, t+1] <- Phi %*% X[, t] + t(rmvnorm(1, c(0, 0), Sigma_eta))
    V <- diag(exp(X[, t+1]/2))
    Y[, t] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
  }
  
  multi_sv_data <- list(X = X, Y = Y, Phi = Phi, Sigma_eta = Sigma_eta, 
                        Sigma_eps = Sigma_eps)
  
  if (save_data) {
    saveRDS(multi_sv_data, file = paste0("./data/bivariate_sv_data_Tfin", Tfin, "_", date, "_", dataset, ".rds"))
  }
} else {
  multi_sv_data <- readRDS(file = paste0("./data/bivariate_sv_data_Tfin", Tfin, "_", date, "_", dataset, ".rds"))
  
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

# par(mfrow = c(1,2))
# hist(Y[1,])
# hist(Y[2,])
############################## Inference #######################################

## Construct initial distribution/prior
prior <- construct_prior(data = Y, prior_type = prior_type)
prior_mean <- prior$prior_mean
prior_var <- prior$prior_var

param_dim <- length(prior_mean)

## Sample from the priors here
# prior_samples <- data.frame(rmvnorm(1000, prior_mean, prior_var))
# names(prior_samples) <- c("phi_11", "phi_12", "phi_21", "phi_22", "sigma_eta1", "sigma_eta2")
prior_samples <- rmvnorm(1000, prior_mean, prior_var)

d <- nrow(Phi)
indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))

### the first 4 elements will be used to construct A
prior_samples_list <- lapply(seq_len(nrow(prior_samples)), function(i) prior_samples[i,])

A_prior_samples <- lapply(prior_samples_list, function(x) matrix(x[1:(d^2)], d, d, byrow = T))

### the last 3 will be used to construct L
construct_Sigma_eta <- function(theta) {
  L <- diag(exp(theta[(d^2+1):param_dim]))
  # L[2,1] <- theta[7]
  # Sigma_eta <- L %*% t(L)
  Sigma_eta <- L
  return(Sigma_eta)
}

Sigma_eta_prior_samples <- lapply(prior_samples_list, construct_Sigma_eta)

## Transform samples of A into samples of Phi via the mapping in Ansley and Kohn (1986)
Phi_prior_samples <- mapply(backward_map, A_prior_samples, Sigma_eta_prior_samples, SIMPLIFY = F)

VAR1_prior_samples <- list()
for (k in 1:nrow(indices)) {
  i <- indices[k, 1]
  j <- indices[k, 2]
  VAR1_prior_samples[[k]] <- unlist(lapply(Phi_prior_samples, function(X) X[i, j]))
}

for (k in 1:d) {
  VAR1_prior_samples[[k + d^2]] <- unlist(lapply(Sigma_eta_prior_samples, function(X) X[k, k]))
}

par(mfrow = c(6, 6))
true_params <- c(t(Phi), diag(Sigma_eta))
param_names <- c("phi_11", "phi_12", "phi_21", "phi_22", "sigma_eta1", "sigma_eta2")
for (i in 1:6) {
  for (j in 1:6) {
    par("mar"=c(4, 4, 2, 2))
    plot(VAR1_prior_samples[[i]], VAR1_prior_samples[[j]], 
         xlab = param_names[i], ylab = param_names[j])
    points(true_params[i], true_params[j], col = "red")
  }
}

if (prior_type == "minnesota") {
  prior_type = ""
} else {
  prior_type = paste0("_", prior_type)
}

## Plot likelihood surface here
if (plot_likelihood_surface) {
  param_grid <- seq(0.1, 0.99, length.out = 100)
  
  k <- seq(-ceiling(Tfin/2)+1, floor(Tfin/2), 1)
  k_in_likelihood <- k [k >= 1 & k <= floor((Tfin-1)/2)]
  freq <- 2 * pi * k_in_likelihood / Tfin
  
  # ## astsa package
  Z <- log(Y^2) - rowMeans(log(Y^2))
  fft_out <- mvspec(t(Z), detrend = F, plot = F)
  # fft_out <- mvspec(t(X), detrend = F, plot = F)
  I_all <- fft_out$fxx
  
  llh <- c()
  for (j in 1:length(param_grid)) {
    Sigma_eta_j <- diag(c(Sigma_eta[1,1], param_grid[j]))
    llh[j] <- compute_whittle_likelihood_multi_sv(Y = Z, fourier_freqs = freq,
                                                  periodogram = I_all,
                                                  params = list(Phi = Phi, Sigma_eta = Sigma_eta_j),
                                                  use_tensorflow = F)$log_likelihood
  }
  
  par(mfrow = c(1,1))
  plot(param_grid, llh, type = "l")
  abline(v = Sigma_eta[2,2], lty = 2)
  abline(v = param_grid[which.max(llh)], col = "red", lty = 2)
  browser()
}


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

####?############################
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

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_Tfin", Tfin, 
                         temper_info, reorder_info, "_", date, "_", dataset, 
                         prior_type, ".rds")

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
                         "_", date, "_", dataset, prior_type, ".rds")

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
                       "_", date, "_", dataset, prior_type, ".rds")


if (rerun_hmc) {
  
  n_post_samples <- 10000
  burn_in <- 1000
  stan.iters <- n_post_samples + burn_in
  
  if (use_heaps_mapping) {
    stan_file <- "./source/stan_multi_sv_heaps.stan"
    multi_sv_data <- list(d = nrow(Y), p = 1, Tfin = ncol(Y), Y = Y,
                          prior_mean_A = prior_mean[c(1,3,2,4)], diag_prior_var_A = diag(prior_var)[c(1,3,2,4)],
                          prior_mean_gamma = prior_mean[5:6], diag_prior_var_gamma = diag(prior_var)[5:6]
    )
  } else {
    stan_file <- "./source/stan_multi_sv.stan"
    multi_sv_data <- list(d = nrow(Y), Tfin = ncol(Y), Y = Y,
                          prior_mean_A = prior_mean[c(1,3,2,4)], diag_prior_var_A = diag(prior_var)[c(1,3,2,4)],
                          prior_mean_gamma = prior_mean[5:6], diag_prior_var_gamma = diag(prior_var)[5:6]
    )
  }
  
  multi_sv_model <- cmdstan_model(
    stan_file,
    cpp_options = list(stan_threads = TRUE)
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

######################################
##   Stan with Whittle likelihood   ##
######################################

hmcw_filepath <- paste0(result_directory, "hmcw_results_Tfin", Tfin, 
                       "_", date, "_", dataset, prior_type, ".rds")

if (rerun_hmcw) {
  print("Starting HMC with Whittle likelihood...")
  
  n_post_samples <- 10000
  burn_in <- 1000
  stan.iters <- n_post_samples + burn_in
  
  stan_file_whittle <- "./source/stan_multi_sv_whittle.stan"
  
  ## Compute periodogram observations here
  ## Fourier frequencies
  k <- seq(-ceiling(Tfin/2)+1, floor(Tfin/2), 1)
  k_in_likelihood <- k [k >= 1 & k <= floor((Tfin-1)/2)]
  freq <- 2 * pi * k_in_likelihood / Tfin
  
  ## astsa package
  Z <- log(Y^2) - rowMeans(log(Y^2))
  fft_out <- mvspec(t(Z), detrend = F, plot = F)
  I <- fft_out$fxx
  
  I_indices <- seq(dim(I)[3])
  I_list <- lapply(I_indices[1:length(freq)], function(x) I[,,x])

  
    
  # periodogram_array <- array(NA, dim = c(length(freq), d, d))
  # for(q in 1:length(freq)) {
  #   periodogram_array[q,,] <- I_list[[q]]
  # }
  
  multi_sv_data_whittle <- list(d = nrow(Y), nfreq = length(freq), freqs = freq,
                                periodogram = periodogram_array,
                                prior_mean_A = prior_mean[c(1,3,2,4)], 
                                diag_prior_var_A = diag(prior_var)[c(1,3,2,4)],
                                prior_mean_gamma = prior_mean[5:6], 
                                diag_prior_var_gamma = diag(prior_var)[5:6]
  )
  
  multi_sv_model_whittle <- cmdstan_model(
    stan_file_whittle,
    cpp_options = list(stan_threads = TRUE)
  )
  
  fit_stan_multi_sv_whittle <- multi_sv_model_whittle$sample(
    multi_sv_data_whittle,
    chains = 1,
    threads = parallel::detectCores(),
    refresh = 5,
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


## Posterior density comparisons

par(mfrow = c(2,4))

plot_margin <- c(-0.2, 0.2)
plot(density(mcmcw.post_samples_phi_11), col = "blue", lty = 2, lwd = 1.5, 
     main = "phi_11", 
     xlim = c(Phi[1,1] + plot_margin))
lines(density(rvgaw.post_samples_phi_11), col = "red", lty = 2, lwd = 1.5,)
lines(density(hmc.post_samples_Phi[,,1]), col = "forestgreen", lwd = 1.5,)
lines(density(hmc.post_samples_Phi[,,1]), col = "goldenrod", lwd = 1.5,)
# lines(density(mcmcw1.post_samples_phi), col = "green")
abline(v = Phi[1,1], lty = 2)
legend("topright", legend = c("MCMCW", "R-VGAW", "HMC"), col = c("blue", "red", "forestgreen"),
       lty = c(2,2,1), cex = 0.7)

plot(density(mcmcw.post_samples_phi_12), col = "blue", lty = 2, lwd = 1.5,
     main = "phi_12", 
     xlim = c(Phi[1,2] + plot_margin))
lines(density(rvgaw.post_samples_phi_12), col = "red", lty = 2, lwd = 1.5,)
lines(density(hmc.post_samples_Phi[,,3]), col = "forestgreen", lwd = 1.5,)
lines(density(hmcw.post_samples_Phi[,,3]), col = "goldenrod", lwd = 1.5,)
abline(v = Phi[1,2], lty = 2)

plot(density(mcmcw.post_samples_phi_21), col = "blue", lty = 2, lwd = 1.5, 
     main = "phi_21", 
     xlim = c(Phi[2,1] + plot_margin))
lines(density(rvgaw.post_samples_phi_21), col = "red", lty = 2, lwd = 1.5)
lines(density(hmc.post_samples_Phi[,,2]), col = "forestgreen", lwd = 1.5)
lines(density(hmcw.post_samples_Phi[,,2]), col = "goldenrod", lwd = 1.5)
abline(v = Phi[2,1], lty = 2)

plot(density(mcmcw.post_samples_phi_22), col = "blue", lty = 2, lwd = 1.5, 
     main = "phi_22", 
     xlim = c(Phi[2,2] + plot_margin))
lines(density(rvgaw.post_samples_phi_22), col = "red", lty = 2, lwd = 1.5)
lines(density(hmc.post_samples_Phi[,,4]), col = "forestgreen", lwd = 1.5)
lines(density(hmcw.post_samples_Phi[,,4]), col = "goldenrod", lwd = 1.5)
abline(v = Phi[2,2], lty = 2)

plot(density(mcmcw.post_samples_sigma_eta_11), col = "blue", lty = 2, lwd = 1.5,
     main = "sigma_eta_11", xlim = c(Sigma_eta[1,1] + c(-0.2, 0.2)))
lines(density(rvgaw.post_samples_sigma_eta_11), col = "red", lty = 2, lwd = 1.5)
lines(density(hmc.post_samples_Sigma_eta[,,1]), col = "forestgreen", lwd = 1.5)
lines(density(hmcw.post_samples_Sigma_eta[,,1]), col = "goldenrod", lwd = 1.5)
# lines(density(mcmcw1.post_samples_sigma_eta2), col = "green")
abline(v = Sigma_eta[1,1], lty = 2)

if (use_cholesky) {
  plot(density(mcmcw.post_samples_sigma_eta_12), col = "blue", lty = 2, lwd = 1.5,
       main = "sigma_eta_12", xlim = c(Sigma_eta[1,2] + c(-0.2, 0.2)))
  lines(density(rvgaw.post_samples_sigma_eta_12), col = "red", lty = 2, lwd = 1.5)
  abline(v = Sigma_eta[1,2], lty = 2)
  
  plot(density(mcmcw.post_samples_sigma_eta_21), col = "blue", lty = 2, lwd = 1.5,
       main = "sigma_eta_21")
  lines(density(rvgaw.post_samples_sigma_eta_21), col = "red", lty = 2, lwd = 1.5)
  abline(v = Sigma_eta[2,1], lty = 2)
}

plot(density(mcmcw.post_samples_sigma_eta_22), col = "blue", lty = 2, lwd = 1.5, 
     main = "sigma_eta_22", xlim = c(Sigma_eta[2,2] + c(-0.2, 0.2)))
lines(density(rvgaw.post_samples_sigma_eta_22), col = "red", lty = 2, lwd = 1.5)
lines(density(hmc.post_samples_Sigma_eta[,,4]), col = "forestgreen", lwd = 1.5)
lines(density(hmcw.post_samples_Sigma_eta[,,4]), col = "goldenrod", lwd = 1.5)
abline(v = Sigma_eta[2,2], lty = 2)


