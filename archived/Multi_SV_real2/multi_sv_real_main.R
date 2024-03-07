## R-VGAW on the SV model

## MCMC on multivariate SV model
rm(list = ls())
setwd("~/R-VGA-Whittle/Multi_SV_real2/")

## Flags
# date <- "20230920" #"20230918" has 5D, "20230920" has 3D
date <- "20240111" #"20230918"
regenerate_data <- F
save_data <- F
use_cholesky <- T # use lower Cholesky factor to parameterise Sigma_eta
transform <- "logit"
prior_type <- "prior1"
use_heaps_mapping <- F
plot_likelihood_surface <- F
plot_prior_samples <- F
plot_trace <- F
plot_trajectories <- F
save_plots <- F

rerun_rvgaw <- T
rerun_mcmcw <- F
rerun_hmc <- T

save_rvgaw_results <- T
save_mcmcw_results <- F
save_hmc_results <- T

## R-VGAW flags
use_tempering <- F
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

#######################
##   Generate data   ##
#######################

dataset <- "hmc_est" #"5" 

Tfin <- 1000
d <- 2L

## Result directory
result_directory <- paste0("./results/", d, "d/")

data <- read.csv("./data/5_Industry_Portfolios_Daily_cleaned.csv")
Y <- data[1:Tfin, 2:3]
Y <- Y - colMeans(Y)
par(mfrow = c(ceiling(d/2),2))
# plot(X[1, ], type = "l")
# plot(X[2, ], type = "l")
# 
for (k in 1:d) {
  plot(Y[, k], type = "l")
}

# par(mfrow = c(1,2))
# hist(Y[1,])
# hist(Y[2,])
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

if (plot_prior_samples) {
  ## Sample from the priors here
  # prior_samples <- data.frame(rmvnorm(1000, prior_mean, prior_var))
  # names(prior_samples) <- c("phi_11", "phi_12", "phi_21", "phi_22", "sigma_eta1", "sigma_eta2")
  prior_samples <- rmvnorm(1000, prior_mean, prior_var)
  
  d <- nrow(Phi)
  indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
  
  ### the first 4 elements will be used to construct A
  prior_samples_list <- lapply(seq_len(nrow(prior_samples)), function(i) prior_samples[i,])
  
  Phi_prior_samples <- lapply(prior_samples_list, function(x) diag(tanh(x[1:d])))
  
  # index_to_i_j_rowwise_diag <- function(k, n) {
  #   p  <- (sqrt(1 + 8 * k) - 1) / 2
  #   i0 <- floor(p)
  #   if (i0 == p) {
  #     return(c(i0, i0)) # (i, j)
  #   } else {
  #     i <- i0 + 1
  #     j <- k - i0 * (i0 + 1) / 2
  #     c(i, j)
  #   }
  # }
  # 
  ### the last 3 will be used to construct L
  construct_Sigma_eta <- function(theta, d) {
    p <- length(theta)
    L <- diag(exp(theta[(d+1):(p-1)]))
    L[2,1] <- theta[p]
    Sigma_eta <- L %*% t(L)
    # Sigma_eta <- L
    return(Sigma_eta)
  }
  
  Sigma_eta_prior_samples <- lapply(prior_samples_list, construct_Sigma_eta, d = d)
  
  ## Transform samples of A into samples of Phi via the mapping in Ansley and Kohn (1986)
  # Phi_prior_samples <- mapply(backward_map, A_prior_samples, Sigma_eta_prior_samples, SIMPLIFY = F)
  
  ## Fix up the indices here
  
  VAR1_prior_samples <- list()
  # for (k in 1:nrow(indices)) {
  #   i <- indices[k, 1]
  #   j <- indices[k, 2]
  #   VAR1_prior_samples[[k]] <- unlist(lapply(Phi_prior_samples, function(X) X[i, j]))
  # }
  
  for (k in 1:d) {
    VAR1_prior_samples[[k]] <- unlist(lapply(Phi_prior_samples, function(X) X[k, k]))
  }
  
  for (k in 1:nrow(indices)) {
    i <- indices[k, 1]
    j <- indices[k, 2]
    VAR1_prior_samples[[d + k]] <- unlist(lapply(Sigma_eta_prior_samples, function(X) X[i, j]))
  }
  
  
  # par(mfrow = c(param_dim, param_dim))
  true_params <- c(diag(Phi), c(t(Sigma_eta)))
  param_names <- c("phi_11", "phi_22", "sigma_eta11", "sigma_eta12", "sigma_eta21", "sigma_eta2")
  
  par(mfrow = c(6,6))
  for (i in 1:6) {
    for (j in 1:6) {
      par("mar"=c(4, 4, 2, 2))
      plot(VAR1_prior_samples[[i]], VAR1_prior_samples[[j]], 
           xlab = param_names[i], ylab = param_names[j])
      points(true_params[i], true_params[j], col = "red", pch = 20)
    }
  }
  
  browser()
}

if (prior_type == "minnesota") {
  prior_type = ""
} else {
  prior_type = paste0("_", prior_type)
}

## Plot likelihood surface here
if (plot_likelihood_surface) {
  print("Plotting likelihood surface...")
  phi_grid <- seq(-0.99, 0.99, length.out = 100)
  sigma_eta_grid <- seq(0.01, 0.99, length.out = 100)
  sigma_eta_grid_offdiag <- seq(0.01, 0.15, length.out = 100)
  
  
  k <- seq(-ceiling(Tfin/2)+1, floor(Tfin/2), 1)
  k_in_likelihood <- k [k >= 1 & k <= floor((Tfin-1)/2)]
  freq <- 2 * pi * k_in_likelihood / Tfin
  
  # ## astsa package
  Z <- log(Y^2) - colMeans(log(Y^2))
  fft_out <- mvspec(Z, detrend = F, plot = F)
  # fft_out <- mvspec(t(X), detrend = F, plot = F)
  I_all <- fft_out$fxx
  # params <- c(t(Phi), diag(Sigma_eta))
  
  llhs <- list()
  # d <- nrow(Phi)
  indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
  
  par(mfrow = c(d,d))
  for (r in 1:d) {
    # i <- indices[r, 1]
    # j <- indices[r, 2]
    
    llh <- c()
    for (q in 1:length(phi_grid)) {
      # Sigma_eta_q <- Sigma_eta
      # Sigma_eta_q[i,j] <- param_grid[q]
      Phi_q <- Phi
      Phi_q[r,r] <- phi_grid[q]
      llh[q] <- compute_whittle_likelihood_multi_sv(Y = Z, fourier_freqs = freq,
                                                    periodogram = I_all,
                                                    params = list(Phi = Phi_q, Sigma_eta = Sigma_eta),
                                                    use_tensorflow = T)$log_likelihood
    }
    
    llhs[[r]] <- llh
    
    param_index <- paste0(r,r)
    plot(phi_grid, llhs[[r]], type = "l", main = bquote(phi[.(param_index)]), 
         ylab = "Log likelihood", xlab = "Parameter")
    abline(v = Phi[r,r], lty = 2)
    abline(v = phi_grid[which.max(llh)], col = "red", lty = 3)
    legend("bottomleft", legend = c("True param", "arg max (llh)"), 
           col = c("black", "red"), lty = c(2,3), cex = 0.5)
  }
  
  for (r in 1:nrow(indices)) {
    i <- indices[r, 1]
    j <- indices[r, 2]
    llh <- c()
    for (q in 1:length(sigma_eta_grid)) {
      Sigma_eta_q <- Sigma_eta
      
      if (i == j) {
        Sigma_eta_q[i,j] <- sigma_eta_grid[q]
      } else {
        Sigma_eta_q[i,j] <- Sigma_eta_q[j,i] <- sigma_eta_grid_offdiag[q]
      }
      
      # Phi_q <- Phi
      # Phi_q[i,j] <- param_grid[j]
      llh[q] <- compute_whittle_likelihood_multi_sv(Y = Z, fourier_freqs = freq,
                                                    periodogram = I_all,
                                                    params = list(Phi = Phi, Sigma_eta = Sigma_eta_q),
                                                    use_tensorflow = F)$log_likelihood
      
    }
    
    llhs[[d+r]] <- llh
    
    param_index <- paste0(i,j)
    if(i == j) {
      plot(sigma_eta_grid, llhs[[d+r]], type = "l", 
           # main = expression(sigma_eta[])
           main = bquote(sigma_eta[.(param_index)]),
           ylab = "Log likelihood", xlab = "Parameter")
      abline(v = (Sigma_eta[i,j]), lty = 2)
      abline(v = sigma_eta_grid[which.max(llh)], col = "red", lty = 3)
      legend("bottomright", legend = c("True param", "arg max (llh)"), 
             col = c("black", "red"), lty = c(2,3), cex = 0.5)
    } else {
      plot(sigma_eta_grid_offdiag, llhs[[d+r]], type = "l", 
           # main = expression(sigma_eta[])
           main = bquote(sigma_eta[.(param_index)]),
           ylab = "Log likelihood", xlab = "Parameter")
      abline(v = (Sigma_eta[i,j]), lty = 2)
      abline(v = sigma_eta_grid_offdiag[which.max(llh)], col = "red", lty = 2)
      legend("bottomright", legend = c("True param", "arg max (llh)"), 
             col = c("black", "red"), lty = c(2,3), cex = 0.5)
      
    }
    
  }
  browser()
  # par(mfrow = c(1,1))
  # plot(param_grid, llh, type = "l", main = "sigma_eta_22", ylab = "Log likelihood", xlab = "Parameter")
  # abline(v = Sigma_eta[2,2], lty = 2)
  # abline(v = param_grid[which.max(llh)], col = "red", lty = 2)
  # legend("topright", legend = c("True param", "arg max (llh)"), col = c("black", "red"), lty = 2)
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
                                      use_cholesky = use_cholesky,
                                      transform = transform,
                                      use_tempering = use_tempering, 
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


browser()
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
                         "_", date, "_", dataset, prior_type, ".rds")

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
                       "_", date, "_", dataset, prior_type, ".rds")


if (rerun_hmc) {
  
  # n_post_samples <- 10000
  # burn_in <- 1000
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

# ## Posterior density comparisons
# 
# par(mfrow = c(2,3))
# # layout.matrix <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3, byrow = T)
# # layout(mat = layout.matrix)
# # # layout.show(5)
# # par(mar = c(4, 4, 4, 4))
# 
# # plot_margin <- c(-0.2, 0.2)
# plot(density(mcmcw.post_samples_phi_11), col = "royalblue", lty = 2, lwd = 2, 
#      main = "phi_11")#, 
# # xlim = c(Phi[1,1] + plot_margin))
# lines(density(rvgaw.post_samples_phi_11), col = "red", lty = 2, lwd = 2)
# lines(density(hmc.post_samples_Phi[,,1]), col = "skyblue", lwd = 2)
# # lines(density(mcmcw1.post_samples_phi), col = "green")
# abline(v = Phi[1,1], lty = 2)
# legend("topright", legend = c("MCMCW", "R-VGAW", "HMC"), col = c("blue", "red", "forestgreen"),
#        lty = c(2,2,1), cex = 0.7)
# 
# # plot(density(mcmcw.post_samples_phi_12), col = "royalblue", lty = 2, lwd = 2,
# #      main = "phi_12", 
# #      xlim = c(Phi[1,2] + plot_margin))
# # lines(density(rvgaw.post_samples_phi_12), col = "red", lty = 2, lwd = 2,)
# # lines(density(hmc.post_samples_Phi[,,3]), col = "skyblue", lwd = 2,)
# # abline(v = Phi[1,2], lty = 2)
# # 
# # plot(density(mcmcw.post_samples_phi_21), col = "royalblue", lty = 2, lwd = 2, 
# #      main = "phi_21", 
# #      xlim = c(Phi[2,1] + plot_margin))
# # lines(density(rvgaw.post_samples_phi_21), col = "red", lty = 2, lwd = 2)
# # lines(density(hmc.post_samples_Phi[,,2]), col = "skyblue", lwd = 2)
# # abline(v = Phi[2,1], lty = 2)
# 
# plot(density(mcmcw.post_samples_phi_22), col = "royalblue", lty = 2, lwd = 2, 
#      main = "phi_22")#, 
# # xlim = c(Phi[2,2] + plot_margin))
# lines(density(rvgaw.post_samples_phi_22), col = "red", lty = 2, lwd = 2)
# lines(density(hmc.post_samples_Phi[,,4]), col = "skyblue", lwd = 2)
# abline(v = Phi[2,2], lty = 2)
# 
# ## Just a plot to print information on tempering, reordering freqs etc.
# plot(x = 0:1,                   # Create empty plot
#      y = 0:1,
#      ann = F,
#      bty = "n",
#      type = "n",
#      xaxt = "n",
#      yaxt = "n")
# text(x = 0.5,                   # Add text to empty plot
#      y = 0.5,
#      paste0("use_tempering = ", use_tempering, "\n reorder_freq = ", reorder_freq),
#      cex = 1)
# 
# 
# plot_margin <- c(-0.1, 0.1)
# indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
# 
# plot(density(mcmcw.post_samples_sigma_eta_11), col = "royalblue", lty = 2, lwd = 2,
#      main = "sigma_eta_11", xlim = c(Sigma_eta[1,1] + plot_margin))
# lines(density(rvgaw.post_samples_sigma_eta_11), col = "red", lty = 2, lwd = 2)
# lines(density(hmc.post_samples_Sigma_eta[,,1]), col = "skyblue", lwd = 2)
# # lines(density(mcmcw1.post_samples_sigma_eta2), col = "green")
# abline(v = Sigma_eta[1,1], lty = 2)
# 
# if (use_cholesky) {
#   plot(density(mcmcw.post_samples_sigma_eta_12), col = "royalblue", lty = 2, lwd = 2,
#        main = "sigma_eta_12", xlim = c(Sigma_eta[1,2] + plot_margin))
#   lines(density(rvgaw.post_samples_sigma_eta_12), col = "red", lty = 2, lwd = 2)
#   lines(density(hmc.post_samples_Sigma_eta[,,3]), col = "skyblue", lwd = 2)
#   abline(v = Sigma_eta[1,2], lty = 2)
#   
#   # plot(density(mcmcw.post_samples_sigma_eta_21), col = "royalblue", lty = 2, lwd = 2,
#   #      main = "sigma_eta_21", xlim = c(Sigma_eta[2,1] + plot_margin))
#   # lines(density(rvgaw.post_samples_sigma_eta_21), col = "red", lty = 2, lwd = 2)
#   # lines(density(hmc.post_samples_Sigma_eta[,,2]), col = "skyblue", lwd = 2)
#   # abline(v = Sigma_eta[2,1], lty = 2)
# }
# 
# plot(density(mcmcw.post_samples_sigma_eta_22), col = "royalblue", lty = 2, lwd = 2,
#      # xlab = "sigma_eta_22",
#      main = "sigma_eta_22", xlim = c(Sigma_eta[2,2] + plot_margin))
# lines(density(rvgaw.post_samples_sigma_eta_22), col = "red", lty = 2, lwd = 2)
# lines(density(hmc.post_samples_Sigma_eta[,,4]), col = "skyblue", lwd = 2)
# abline(v = Sigma_eta[2,2], lty = 2)

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
       main = bquote(phi[.(ind)]))
  lines(density(mcmcw.post_samples_phi), col = "royalblue", lty = 3, lwd = 2)
  # lines(density(hmc.post_samples_phi), col = "skyblue")
  
  lines(density(hmc.post_samples_Phi[,,hmc_indices[k]]), col = "skyblue", lty = 1, lwd = 2)
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
       main = bquote(sigma_eta[.(ind)]))
  lines(density(mcmcw.post_samples_sigma_eta), col = "royalblue", lty = 3, lwd = 2)
  lines(density(hmc.post_samples_Sigma_eta[,,hmc_indices[k]]), lty = 1, col = "skyblue", lwd = 2)
  # legend("topright", legend = c("R-VGAW", "HMC"), col = c("red", "forestgreen"),
  #        lty = c(2,2,1), cex = 0.3, y.intersp = 0.25)
}

if (plot_trace) { # for mcmcw
  
  thinning_interval <- seq(1, iters, by = 1)
  for (k in 1:d) {
    mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[k,k]))
    ## red = R-VGA Whittle, blue = MCMC Whittle, green = HMC
    
    mcmcw.post_samples_phi <- mcmcw.post_samples_phi[-(1:burn_in)]
    mcmcw.post_samples_phi_thinned <- mcmcw.post_samples_phi[thinning_interval]                      
    coda::traceplot(coda::as.mcmc(mcmcw.post_samples_phi_thinned))
  }
  
  for (k in 1:nrow(indices)) {
    i <- indices[k, 1]
    j <- indices[k, 2]
    
    mcmcw.post_samples_sigma_eta <- unlist(lapply(mcmcw.post_samples_Sigma_eta, function(x) x[i,j]))
    mcmcw.post_samples_sigma_eta <- mcmcw.post_samples_sigma_eta[-(1:burn_in)]
    mcmcw.post_samples_sigma_eta_thinned <- mcmcw.post_samples_sigma_eta[thinning_interval]
    coda::traceplot(coda::as.mcmc(mcmcw.post_samples_sigma_eta_thinned))
  }
  
}


if (plot_trajectories) {
  ## Parameter trajectories
  par(mfrow = c(2,3))
  trajectories <- list()
  for (p in 1:param_dim) {
    trajectories[[p]] <- sapply(rvgaw_results$mu, function(e) e[p])
    plot(trajectories[[p]], type = "l", xlab = "Iteration", ylab = "param", main = "")
  }
}

if (save_plots) {
  plot_file <- paste0("sv_posterior_", d, "d", temper_info, reorder_info,
                      "_", date, ".png")
  filepath = paste0("./plots/", plot_file)
  png(filepath, width = 1000, height = 1000)
  
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
    # lines(density(hmc.post_samples_phi), col = "skyblue")
    
    lines(density(hmc.post_samples_Phi[,,hmc_indices[k]]), col = "skyblue", lty = 1, lwd = 2)
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
    lines(density(hmc.post_samples_Sigma_eta[,,hmc_indices[k]]), lty = 1, col = "skyblue", lwd = 2)
    abline(v = Sigma_eta[i,j], lty = 2, lwd = 2)
    # legend("topright", legend = c("R-VGAW", "HMC"), col = c("red", "forestgreen"),
    #        lty = c(2,2,1), cex = 0.3, y.intersp = 0.25)
  }
  
  dev.off()
}
