## R-VGAW on the SV model

## MCMC on multivariate SV model
rm(list = ls())
setwd("~/R-VGA-Whittle/Multi_SV_real/")

library("coda")
library("mvtnorm")
library("astsa")
library("cmdstanr")
# library("expm")
library("stcos")
library(dplyr)
reticulate::use_condaenv("myenv", required = TRUE)
library(tensorflow)
tfp <- import("tensorflow_probability")
tfd <- tfp$distributions
library(keras)
library(Matrix)

source("./source/run_rvgaw_multi_sv.R")
source("./source/run_mcmc_multi_sv.R")
source("./source/compute_whittle_likelihood_multi_sv.R")
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

## Flags
date <- "20240115_agricfood" #"20230918"
use_cholesky <- T # use lower Cholesky factor to parameterise Sigma_eta
# prior_type <- "minnesota"
use_heaps_mapping <- F
plot_likelihood_surface <- F
plot_prior_samples <- F
plot_trajectories <- F
plot_trace <- T
transform <- "arctanh"
prior_type <- "prior1"

# reread_data <- F
rerun_rvgaw <- T
rerun_mcmcw <- F
rerun_hmc <- F

save_rvgaw_results <- T
save_mcmcw_results <- F
save_hmc_results <- F
save_plots <- F

## R-VGAW flags
use_tempering <- F
reorder_freq <- T
decreasing <- F
use_median <- F

#########################
##      Read data      ##
#########################

dataset <- "daily" # daily or monthly
industries <- c("Agric", "Food")
nstocks <- 2
nobs <- 1000

# result_directory <- paste0("./results/", dataset, "_", nstocks, "stocks/")
result_directory <- paste0("./results/49_industries/", dataset, "_", nstocks, "stocks/", transform, "/")

# if (reread_data) {
  if (dataset == "daily") {
    # returns_data <- read.csv("./data/5_Industry_Portfolios_Daily_cleaned.csv")
    returns_data <- read.csv("./data/49_Industry_Portfolios_Daily.csv")
    
    datafile <- paste0("_daily", nobs)
    # Y <- returns_data[1:nobs, 2:(2+nstocks-1)]
    # Y <- returns_data[c("Toys", "Books")]
    # Y <- returns_data[c("Agric", "BusSv")][1:nobs, ]
    # Y <- returns_data[c("MedEq", "BusSv")][1:nobs, ]
    # Y <- returns_data[c("Rtail", "Food")][1:nobs, ]
    Y <- returns_data[industries][1:nobs, ]
    
  } else { # monthly
    returns_data <- read.csv("./data/5_Industry_Portfolios_cleaned.CSV")
    datafile <- paste0("_monthly", nobs)
    Y <- returns_data[1:nobs, 2:(2+nstocks-1)]
  }
  
#   ## Save data?
#   saveRDS(Y, paste0("./data/data_", paste(industries, collapse = ""), "_", date))
# } else {
#   Y <- readRDS(paste0("./data/data_", paste(industries, collapse = ""), "_", date))
# }

# library("dplyr")
# test <- returns_data %>% select_if(!any(.) == -99.99)

all_data <- returns_data[1:nobs, c("Agric", "Food", "Beer",  "Smoke", 
                          "Toys", "Fun", "Books", "Hshld", "Clths", "MedEq",
                          "Drugs", "Chems", "Txtls", "BldMt", "Cnstr", "Steel", 
                          "Mach", "ElcEq", "Autos", "Aero", "Ships", "Mines", 
                          "Coal", "Oil", "Util", "Telcm", "BusSv", "Hardw", 
                          "Chips", "LabEq", "Boxes", "Trans", "Whlsl", "Rtail", 
                          "Meals", "Banks", "Insur", "RlEst", "Fin", "Other")]
all_data <- all_data - colMeans(all_data)
cormat <- cor(all_data)

# Agric and Toys have low correlation (1000 obs)
# Toys and BusSv have low correlation (2000 obs)

# Y <- returns_data[, 2:4]
# Y_demeaned <- Y - colMeans(Y)
Y_demeaned <- Y
for (col in 1:ncol(Y)) {
  Y_demeaned[, col] <- Y[, col] - mean(Y[, col])
}

d <- ncol(Y_demeaned)
# par(mfrow = c(d, 1))
# for (c in 1:ncol(Y_demeaned)) {
#   plot(Y_demeaned[, c], type = "l")
# }

############################## Inference #######################################

## Change prior to new set of parameters -- maybe just put priors on Phi_11, Phi_22
## Constrain so that Phi_11 and Phi_22 are both in (-1,1) -- use arctanh() for this?
## Parameterise Sigma_eta = LL^T and put prior on LL^T


## Construct initial distribution/prior
prior <- construct_prior(data = Y_demeaned, use_cholesky = use_cholesky, prior_type = prior_type)
prior_mean <- prior$prior_mean
prior_var <- prior$prior_var

param_dim <- length(prior_mean)

if (plot_prior_samples) {
  print("Plotting prior samples...")
  ## Sample from the priors here
  # prior_samples <- data.frame(rmvnorm(1000, prior_mean, prior_var))
  # names(prior_samples) <- c("phi_11", "phi_12", "phi_21", "phi_22", "sigma_eta1", "sigma_eta2")
  prior_samples <- rmvnorm(10000, prior_mean, prior_var)
  
  # d <- nrow(Phi)
  #indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
  indices <- data.frame(i = c(1,2,2), j = c(1,2,1))

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
  construct_Sigma_eta2 <- function(theta, d) {
    p <- length(theta)
    L <- diag(exp(theta[(d+1):(p-1)]))
    L[2,1] <- theta[p]
    Sigma_eta <- L %*% t(L)
    # Sigma_eta <- L
    return(Sigma_eta)
  }
  
  # Sigma_eta_prior_samples <- lapply(prior_samples_list, construct_Sigma_eta, d = d)
  
  Sigma_eta_prior_samples <- lapply(prior_samples_list, construct_Sigma_eta2, d = d)
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
  # true_params <- c(diag(Phi), c(t(Sigma_eta)))
  param_names <- c("phi_11", "phi_22", "sigma_eta11", "sigma_eta22", "sigma_eta21")
  
  par(mfrow = c(param_dim, param_dim))
  for (i in 1:param_dim) {
    for (j in 1:param_dim) {
      par("mar"=c(4, 4, 2, 2))
      plot(VAR1_prior_samples[[i]], VAR1_prior_samples[[j]], 
           xlab = param_names[i], ylab = param_names[j])
      # points(true_params[i], true_params[j], col = "red", pch = 20)
    }
  }
  
  browser()
}

# if (prior_type == "minnesota") {
#   prior_type = ""
# } else {
#   prior_type = paste0("_", prior_type)
# }

## Plot likelihood surface here
if (plot_likelihood_surface) {
  print("Plotting likelihood surface...")
  phi_grid <- seq(-0.99, 0.99, length.out = 100)
  sigma_eta_grid <- seq(0.01, 0.99, length.out = 100)
  sigma_eta_grid_offdiag <- seq(0.01, 0.15, length.out = 100)
  
  Tfin <- length(Y_demeaned)
  k <- seq(-ceiling(Tfin/2)+1, floor(Tfin/2), 1)
  k_in_likelihood <- k [k >= 1 & k <= floor((Tfin-1)/2)]
  freq <- 2 * pi * k_in_likelihood / Tfin
  
  # ## astsa package
  Z <- log(Y_demeaned^2) - rowMeans(log(Y_demeaned^2))
  fft_out <- mvspec(t(Z), detrend = F, plot = F)
  # fft_out <- mvspec(t(X), detrend = F, plot = F)
  I_all <- fft_out$fxx

  # params <- c(t(Phi), diag(Sigma_eta))
  ## Plot the spectral density
  # compute_whittle_likelihood_multi_sv(Y = Z, fourier_freqs = freq,
  #                                     periodogram = I_all,
  #                                     params = list(Phi = Phi, Sigma_eta = Sigma_eta),
  #                                     use_tensorflow = T)$log_likelihood
  browser()

  llhs <- list()
  d <- nrow(Phi)
  indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
  
  par(mfrow = c(4,2))
  for (r in 1:nrow(indices)) {
    i <- indices[r, 1]
    j <- indices[r, 2]
    
    llh <- c()
    for (q in 1:length(phi_grid)) {
      # Sigma_eta_q <- Sigma_eta
      # Sigma_eta_q[i,j] <- param_grid[q]
      Phi_q <- Phi
      Phi_q[i,j] <- phi_grid[q]
      llh[q] <- compute_whittle_likelihood_multi_sv(Y = Z, fourier_freqs = freq,
                                                    periodogram = I_all,
                                                    params = list(Phi = Phi_q, Sigma_eta = Sigma_eta),
                                                    use_tensorflow = T)$log_likelihood
    }
    
    llhs[[r]] <- llh
    
    param_index <- paste0(i,j)
    plot(phi_grid, llhs[[r]], type = "l", main = bquote(phi[.(param_index)]), 
         ylab = "Log likelihood", xlab = "Parameter")
    abline(v = Phi[i,j], lty = 2)
    abline(v = phi_grid[which.max(llh)], col = "red", lty = 3)
    legend("bottomright", legend = c("True param", "arg max (llh)"), 
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
    
    llhs[[r+d^2]] <- llh
    
    param_index <- paste0(i,j)
    if(i == j) {
      plot(sigma_eta_grid, llhs[[r+d^2]], type = "l", 
           # main = expression(sigma_eta[])
           main = bquote(sigma_eta[.(param_index)]),
           ylab = "Log likelihood", xlab = "Parameter")
      abline(v = (Sigma_eta[i,j]), lty = 2)
      abline(v = sigma_eta_grid[which.max(llh)], col = "red", lty = 3)
      legend("bottomright", legend = c("True param", "arg max (llh)"), 
             col = c("black", "red"), lty = c(2,3), cex = 0.5)
    } else {
      plot(sigma_eta_grid_offdiag, llhs[[r+d^2]], type = "l", 
           # main = expression(sigma_eta[])
           main = bquote(sigma_eta[.(param_index)]),
           ylab = "Log likelihood", xlab = "Parameter")
      abline(v = (Sigma_eta[i,j]), lty = 2)
      abline(v = sigma_eta_grid_offdiag[which.max(llh)], col = "red", lty = 2)
      legend("bottomright", legend = c("True param", "arg max (llh)"), 
             col = c("black", "red"), lty = c(2,3), cex = 0.5)
      
    }
    
  }
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

S <- 5000L
a_vals <- 1

################ R-VGA starts here #################
print("Starting R-VGAL with Whittle likelihood...")

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_realdata",  
                         datafile, temper_info, reorder_info, "_", date, ".rds")

if (rerun_rvgaw) {
  rvgaw_results <- run_rvgaw_multi_sv(data = Y_demeaned, prior_mean = prior_mean, 
                                      prior_var = prior_var, S = S,
                                      use_cholesky = use_cholesky,
                                      use_tempering = use_tempering, 
                                      temper_schedule = temper_schedule, 
                                      reorder_freq = reorder_freq, 
                                      decreasing = decreasing,
                                      transform = transform,
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
# 
# par(mfrow = c(3,2))
# plot(density(rvgaw.post_samples_phi_11), col = "royalblue", main = "phi_11")
# plot(density(rvgaw.post_samples_phi_22), col = "royalblue", main = "phi_22")
# 
# plot(density(rvgaw.post_samples_sigma_eta_11), col = "royalblue", main = "sigma_eta_11")
# 
# if (use_cholesky) {
#   plot(density(rvgaw.post_samples_sigma_eta_12), col = "royalblue", main = "sigma_eta_12")
#   plot(density(rvgaw.post_samples_sigma_eta_21), col = "royalblue", main = "sigma_eta_21")
# }
# 
# plot(density(rvgaw.post_samples_sigma_eta_22), col = "royalblue", main = "sigma_eta_22")

#############################
##   MCMC implementation   ##
#############################
print("Starting MCMC with Whittle likelihood...")

mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_results_realdata", 
                         datafile, "_", date, ".rds")

n_post_samples <- 10000
burn_in <- 10000
# n_chains <- 2
iters <- n_post_samples + burn_in

if (rerun_mcmcw) {
  mcmcw_results <- run_mcmc_multi_sv(data = Y_demeaned, iters = iters, burn_in = burn_in, 
                                     prior_mean = prior_mean, prior_var = prior_var,
                                     adapt_proposal = T, use_whittle_likelihood = T,
                                     use_cholesky = use_cholesky,
                                     transform = transform)
  if (save_mcmcw_results) {
    saveRDS(mcmcw_results, mcmcw_filepath)
  }
} else {
  mcmcw_results <- readRDS(mcmcw_filepath)
}

## Extract samples
mcmcw.post_samples_Phi <- lapply(mcmcw_results$post_samples, function(x) x$Phi) #post_samples_theta[, 1]
mcmcw.post_samples_Sigma_eta <- lapply(mcmcw_results$post_samples, function(x) x$Sigma_eta) #post_samples_theta[, 2]

# mcmcw.post_samples_phi_11 <- lapply(mcmcw.post_samples_phi, function(x) x[1,1])
# mcmcw.post_samples_phi_12 <- lapply(mcmcw.post_samples_phi, function(x) x[1,2])
# mcmcw.post_samples_phi_21 <- lapply(mcmcw.post_samples_phi, function(x) x[2,1])
# mcmcw.post_samples_phi_22 <- lapply(mcmcw.post_samples_phi, function(x) x[2,2])
# 
# mcmcw.post_samples_phi_11 <- as.mcmc(unlist(mcmcw.post_samples_phi_11[-(1:burn_in)]))
# mcmcw.post_samples_phi_12 <- as.mcmc(unlist(mcmcw.post_samples_phi_12[-(1:burn_in)]))
# mcmcw.post_samples_phi_21 <- as.mcmc(unlist(mcmcw.post_samples_phi_21[-(1:burn_in)]))
# mcmcw.post_samples_phi_22 <- as.mcmc(unlist(mcmcw.post_samples_phi_22[-(1:burn_in)]))
# 
# mcmcw.post_samples_sigma_eta_11 <- lapply(mcmcw.post_samples_sigma_eta, function(x) x[1,1])
# mcmcw.post_samples_sigma_eta_12 <- lapply(mcmcw.post_samples_sigma_eta, function(x) x[1,2])
# mcmcw.post_samples_sigma_eta_21 <- lapply(mcmcw.post_samples_sigma_eta, function(x) x[2,1])
# mcmcw.post_samples_sigma_eta_22 <- lapply(mcmcw.post_samples_sigma_eta, function(x) x[2,2])
# 
# mcmcw.post_samples_sigma_eta_11 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_11[-(1:burn_in)]))
# mcmcw.post_samples_sigma_eta_12 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_12[-(1:burn_in)]))
# mcmcw.post_samples_sigma_eta_21 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_21[-(1:burn_in)]))
# mcmcw.post_samples_sigma_eta_22 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_22[-(1:burn_in)]))
# 
# 
# par(mfrow = c(3,2))
# coda::traceplot(mcmcw.post_samples_phi_11, main = "Trace plot for phi_11")
# coda::traceplot(mcmcw.post_samples_phi_22, main = "Trace plot for phi_22")
# coda::traceplot(mcmcw.post_samples_sigma_eta_11, main = "Trace plot for sigma_eta_11")
# 
# if (use_cholesky) {
#   coda::traceplot(mcmcw.post_samples_sigma_eta_12, main = "Trace plot for sigma_eta_12")
#   coda::traceplot(mcmcw.post_samples_sigma_eta_21, main = "Trace plot for sigma_eta_21")
# }
# coda::traceplot(mcmcw.post_samples_sigma_eta_22, main = "Trace plot for sigma_eta_22")

########################
###       STAN       ###
########################
print("Starting HMC...")

hmc_filepath <- paste0(result_directory, "hmc_results_realdata", 
                       datafile, "_", date, ".rds")


if (rerun_hmc) {
  
  n_post_samples <- 10000
  burn_in <- 5000
  n_chains <- 2
  stan.iters <- n_post_samples + burn_in
  d <- as.integer(ncol(Y_demeaned))
  
  use_chol <- 0
  if (use_cholesky) {
    use_chol <- 1
  }
  
  stan_file <- "./source/stan_multi_sv.stan"
  multi_sv_data <- list(d = d, Tfin = nrow(Y), Y = Y_demeaned,
                        prior_mean_Phi = prior_mean[1:d], diag_prior_var_Phi = diag(prior_var)[1:d],
                        prior_mean_gamma = prior_mean[(d+1):param_dim], diag_prior_var_gamma = diag(prior_var)[(d+1):param_dim],
                        use_chol = 0, transform = ifelse(transform == "arctanh", 1, 0))
    
    
  multi_sv_model <- cmdstan_model(
    stan_file,
    cpp_options = list(stan_threads = TRUE)
  )
  
  fit_stan_multi_sv <- multi_sv_model$sample(
    multi_sv_data,
    chains = n_chains,
    threads = parallel::detectCores(),
    refresh = 100,
    iter_warmup = burn_in,
    iter_sampling = n_post_samples
  )
  
  stan_results <- list(draws = fit_stan_multi_sv$draws(variables = c("Phi_mat", "Sigma_eta_mat")),
                       time = fit_stan_multi_sv$time)
  
  if (save_hmc_results) {
    saveRDS(stan_results, hmc_filepath)
  }
  
} else {
  # stan_results <- readRDS(hmc_filepath)
}

# hmc.post_samples_Phi <- stan_results$draws[,,1:(d^2)]
# hmc.post_samples_Sigma_eta <- stan_results$draws[,,(d^2+1):(2*d^2)]

######################################
##   Stan with Whittle likelihood   ##
######################################

hmcw_filepath <- paste0(result_directory, "hmcw_results_realdata",  
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
  Z <- log(Y_demeaned^2) - colMeans(log(Y_demeaned^2))
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

### Posterior of diagonal entries of Phi  
for (k in 1:d) {    
  rvgaw.post_samples_phi <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[k,k]))
  mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[k,k]))
  
  ind <- paste0(k,k)
  plot(density(rvgaw.post_samples_phi), col = "red", lty = 2, lwd = 2,
       main = bquote(phi[.(ind)]), xlim = c(0.1, 0.99))
  lines(density(mcmcw.post_samples_phi), col = "royalblue", lty = 2, lwd = 2)
  lines(density(hmcw.post_samples_Phi[,,hmc_indices[k]]), col = "deepskyblue")
  # legend("topleft", legend = c("R-VGA Whittle", "MCMC Whittle", "HMC"), 
  #        col = c("red", "royalblue", "deepskyblue"),
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
  plot(density(rvgaw.post_samples_sigma_eta), col = "red", lty = 2, lwd = 2,
    main = bquote(sigma_eta[.(ind)]), xlim = c(0, 2))
  lines(density(mcmcw.post_samples_sigma_eta), col = "royalblue", lty = 2, lwd = 2)
  lines(density(hmcw.post_samples_Sigma_eta[,,hmc_indices[k]]), col = "deepskyblue")
  # abline(v = Sigma_eta[i,j], lty = 2)
  # legend("topright", legend = c("R-VGAW", "HMC"), col = c("red", "forestgreen"),
  #        lty = c(2,2,1), cex = 0.3, y.intersp = 0.25)
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

if (plot_trace) { # for mcmcw
  par(mfrow = c(ceiling(d+d^2)/2, 2))
  thinning_interval <- seq(1, iters, by = 1)
  for (k in 1:d) {
    mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[k,k]))
    ## red = R-VGA Whittle, blue = MCMC Whittle, green = HMC
    
    mcmcw.post_samples_phi <- mcmcw.post_samples_phi[-(1:burn_in)]
    # mcmcw.post_samples_phi_thinned <- mcmcw.post_samples_phi[thinning_interval]                      
    coda::traceplot(coda::as.mcmc(mcmcw.post_samples_phi), main = "phi")
  }
  
  for (k in 1:nrow(indices)) {
    i <- indices[k, 1]
    j <- indices[k, 2]
    
    mcmcw.post_samples_sigma_eta <- unlist(lapply(mcmcw.post_samples_Sigma_eta, function(x) x[i,j]))
    mcmcw.post_samples_sigma_eta <- mcmcw.post_samples_sigma_eta[-(1:burn_in)]
    # mcmcw.post_samples_sigma_eta_thinned <- mcmcw.post_samples_sigma_eta[thinning_interval]
    coda::traceplot(coda::as.mcmc(mcmcw.post_samples_sigma_eta), main = "sigma_eta")
  }
  
  ## HMC trace plots
  # hmc.post_samples_phi11 <- as.vector(hmc.post_samples_Phi[,,1])
  # hmc.post_samples_phi22 <- as.vector(hmc.post_samples_Phi[,,4])
  # coda::traceplot(coda::as.mcmc(hmc.post_samples_phi11), ylab = "phi_11")
  # coda::traceplot(coda::as.mcmc(hmc.post_samples_phi22), ylab = "phi_22")
  # 
  # hmc.post_samples_sigma_eta11 <- as.vector(hmc.post_samples_Sigma_eta[,,1])
  # hmc.post_samples_sigma_eta21 <- as.vector(hmc.post_samples_Sigma_eta[,,2])
  # hmc.post_samples_sigma_eta22 <- as.vector(hmc.post_samples_Sigma_eta[,,4])
  # 
  # coda::traceplot(coda::as.mcmc(hmc.post_samples_sigma_eta11), ylab = "sigma_eta_11")
  # coda::traceplot(coda::as.mcmc(hmc.post_samples_sigma_eta21), ylab = "sigma_eta_21")
  # coda::traceplot(coda::as.mcmc(hmc.post_samples_sigma_eta22), ylab = "sigma_eta_22")
  
}

if (save_plots) {
  plot_file <- paste0("svreal_posterior_", d, "d", "_", nobs, "obs", temper_info, reorder_info,
                      "_", date, ".png")
  filepath = paste0("./plots/", plot_file)
  png(filepath, width = 600, height = 800)
  
  ## Plot posterior estimates
  indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
  hmc_indices <- diag(matrix(1:d^2, d, d, byrow = T))
  par(mfrow = c(d+1,d))
  
  ### Posterior of diagonal entries of Phi  
  for (k in 1:d) {    
  rvgaw.post_samples_phi <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[k,k]))
  mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[k,k]))
  
  ind <- paste0(k,k)
  plot(density(rvgaw.post_samples_phi), col = "red", lty = 2, lwd = 2,
       main = bquote(phi[.(ind)]), xlim = c(0.1, 0.99))
  lines(density(mcmcw.post_samples_phi), col = "royalblue", lty = 2, lwd = 2)
  lines(density(hmc.post_samples_Phi[,,hmc_indices[k]]), col = "deepskyblue")
  # legend("topleft", legend = c("R-VGA Whittle", "MCMC Whittle", "HMC"), 
  #        col = c("red", "royalblue", "deepskyblue"),
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
    plot(density(rvgaw.post_samples_sigma_eta), col = "red", lty = 2, lwd = 2,
         main = bquote(sigma_eta[.(ind)]), xlim = c(0.1, 0.7))
    lines(density(mcmcw.post_samples_sigma_eta), col = "royalblue", lty = 2, lwd = 2)
    lines(density(hmc.post_samples_Sigma_eta[,,hmc_indices[k]]), lwd = 2, col = "deepskyblue")
    # legend("topright", legend = c("R-VGAW", "HMC"), col = c("red", "forestgreen"),
    #        lty = c(2,2,1), cex = 0.3, y.intersp = 0.25)
  }
  
  dev.off()
  
}
