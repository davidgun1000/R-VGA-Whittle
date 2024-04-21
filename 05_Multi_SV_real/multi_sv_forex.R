## R-VGAW on the SV model

## MCMC on multivariate SV model
rm(list = ls())
setwd("~/R-VGA-Whittle/05_Multi_SV_real/")

library("coda")
library("mvtnorm")
library("astsa")
library("cmdstanr")
library(dplyr)
# library("expm")
# library("stcos")
# library(dplyr)
reticulate::use_condaenv("myenv", required = TRUE)
library(tensorflow)
tfp <- import("tensorflow_probability")
tfd <- tfp$distributions
library(keras)
library(Matrix)
library(ggplot2)
library(grid)
library(gridExtra)
library(gtable)


# source("./source/run_rvgaw_multi_sv.R")
source("./source/run_rvgaw_multi_sv_block.R")
source("./source/run_mcmc_multi_sv.R")
source("./source/compute_whittle_likelihood_multi_sv.R")
source("./source/construct_prior2.R")
source("./source/map_functions.R")
source("./source/construct_Sigma.R")
# source("./archived/compute_partial_whittle_likelihood.R")
# source("./source/compute_grad_hessian.R")
source("./source/compute_grad_hessian_block.R")
source("./source/compute_periodogram.R")


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
date <- "20240115" #"20230918"
use_cholesky <- T # use lower Cholesky factor to parameterise Sigma_eta
# prior_type <- "minnesota"
use_heaps_mapping <- F
plot_likelihood_surface <- F
plot_prior_samples <- T
plot_trajectories <- F
plot_trace <- F
transform <- "arctanh"
prior_type <- "prior1"
currencies <- c("GBP", "USD")
# reread_data <- F
rerun_rvgaw <- F
rerun_mcmcw <- F
rerun_hmc <- F
rerun_hmcw <- F

save_rvgaw_results <- F
save_mcmcw_results <- F
save_hmc_results <- F
save_hmcw_results <- F

save_plots <- T

## R-VGAW flags
use_tempering <- T
temper_first <- T
reorder <- 0 #"decreasing"
reorder_seed <- 2024
# decreasing <- F
use_median <- F
# nblocks <- 100
blocksize <- 500
n_indiv <- 100

## HMC flags
n_post_samples <- 10000
burn_in <- 5000
n_chains <- 2

#########################
##      Read data      ##
#########################

# dataset <- "daily" # daily or monthly
nstocks <- 2
# nobs <- 1000

# result_directory <- paste0("./results/", dataset, "_", nstocks, "stocks/")
result_directory <- paste0("./results/forex/", nstocks, "d/", transform, "/")

## Exchange rate data
load("./data/exrates.RData")

log_data <- mutate_all(dat, function(x) c(0, log(x[2:length(x)] / x[1:(length(x)-1)]) * 100))

exrates <- log_data[-1, ] # get rid of 1st row
# Y <- exrates[, 1:nstocks]
Y <- exrates[, currencies]

# data <- dat[, c("AUD", "NZD", "USD")]
# nrows <- nrow(data)

# # Compute log returns
# data$AUD_returns <- c(0, log(data$AUD[2:nrows] / data$AUD[1:(nrows-1)])*100)
# data$NZD_returns <- c(0, log(data$NZD[2:nrows] / data$NZD[1:(nrows-1)])*100)
# data$USD_returns <- c(0, log(data$USD[2:nrows] / data$USD[1:(nrows-1)])*100)

# exrates <- data[-1, c("AUD_returns", "NZD_returns", "USD_returns")] # get rid of 1st row
# Y <- exrates[, 1:nstocks]
# Y <- exrates[, c("AUD_returns", "USD_returns")]
# eur_usd <- read.csv("./data/EURUSD.csv")
# aud_usd <- read.csv("./data/AUDUSD.csv")
# nzd_usd <- read.csv("./data/NZDUSD.csv")
# 
# aud_usd_sub <- aud_usd %>% select(Date, Adj.Close)
# eur_usd_sub <- eur_usd %>% select(Date, Adj.Close) %>%
#                 filter(Date >= "2006-05-16")
# 
# nzd_usd_sub <- nzd_usd %>% select(Date, Adj.Close) %>%
#                 filter(Date >= "2006-05-16")
# 
# df <- inner_join(eur_usd_sub, aud_usd_sub, by = "Date")
# df2 <- inner_join(df, nzd_usd_sub, by = "Date")
# names(df2) <- c("Date", "EURUSD", "AUDUSD", "NZDUSD")

Y_demeaned <- Y - colMeans(Y)
# Y_std2 <- (Y - colMeans(Y))/apply(Y, 2, sd)
d <- ncol(Y_demeaned)

par(mfrow = c(d, 1))
for (c in 1:ncol(Y_demeaned)) {
  plot(Y_demeaned[, c], type = "l")
}

#########
##################### Inference #######################################

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
  
  # browser()
}

## Plot likelihood surface here
if (plot_likelihood_surface) {
  print("Plotting likelihood surface...")
  phi_grid <- seq(-0.99, 0.99, length.out = 100)
  sigma_eta_grid <- seq(0.01, 0.99, length.out = 100)
  sigma_eta_grid_offdiag <- seq(0.01, 0.15, length.out = 100)
  
  Tfin <- nrow(Y_demeaned)
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

####?###########################
##    R-VGAW implementation   ##
################################

if (use_tempering) {
  n_temper <- 5
  K <- 100
  temper_schedule <- rep(1/K, K)
  temper_info <- paste0("_temper", n_temper)
} else {
  temper_info <- ""
}

if (reorder == "random") {
  reorder_info <- paste0("_", reorder, reorder_seed)
} else if (reorder == "decreasing") {
  reorder_info <- paste0("_", reorder)
} else if (reorder > 0) {
  reorder_info <- paste0("_reorder", reorder)
} else {
  reorder_info <- ""
}

# if (!is.null(nblocks)) {
if (!is.null(blocksize)) {
  # block_info <- paste0("_", nblocks, "blocks", n_indiv, "indiv")
  block_info <- paste0("_", "blocksize", blocksize, "_", n_indiv, "indiv")
} else {
  block_info <- ""
}

S <- 1000L
a_vals <- 1

################ R-VGA starts here #################
print("Starting R-VGAL with Whittle likelihood...")

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_forex", 
                         "_", paste(currencies, collapse = "_"), 
                         temper_info, reorder_info, block_info, "_", date, ".rds")

if (rerun_rvgaw) {
  rvgaw_results <- run_rvgaw_multi_sv(data = Y_demeaned, prior_mean = prior_mean, 
                                      prior_var = prior_var, S = S,
                                      use_cholesky = use_cholesky,
                                      use_tempering = use_tempering, 
                                      temper_schedule = temper_schedule, 
                                      temper_first = temper_first,
                                      reorder = reorder, 
                                      # decreasing = decreasing,
                                      reorder_seed = reorder_seed,
                                      # nblocks = nblocks,
                                      blocksize = blocksize,
                                      n_indiv = n_indiv)
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

mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_forex", 
                         "_", date, ".rds")

# n_post_samples <- 10000
# burn_in <- 10000
# n_chains <- 2
iters <- (n_post_samples + burn_in) * n_chains

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

hmc_filepath <- paste0(result_directory, "hmc_forex", 
                      "_", paste(currencies, collapse = "_"), 
                       "_", date, ".rds")


if (rerun_hmc) {
  
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
  
  hmc_results <- list(draws = fit_stan_multi_sv$draws(variables = c("Phi_mat", "Sigma_eta_mat")),
                       time = fit_stan_multi_sv$time)
  
  if (save_hmc_results) {
    saveRDS(hmc_results, hmc_filepath)
  }
  
} else {
  hmc_results <- readRDS(hmc_filepath)
}

hmc.post_samples_Phi <- hmc_results$draws[,,1:(d^2)]
hmc.post_samples_Sigma_eta <- hmc_results$draws[,,(d^2+1):(2*d^2)]

######################################
##   Stan with Whittle likelihood   ##
######################################

hmcw_filepath <- paste0(result_directory, "hmcw_forex", 
                        "_", paste(currencies, collapse = "_"),
                         "_", date, ".rds")

if (rerun_hmcw) {
  print("Starting HMC with Whittle likelihood...")
  
  stan_file_whittle <- "./source/stan_multi_sv_whittle.stan"
  
  # ## Calculation of Whittle likelihood
  ## Fourier frequencies
  Tfin <- nrow(Y_demeaned)
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
  multi_sv_data_whittle <- list(d = ncol(Y_demeaned), nfreq = length(freq), freqs = freq,
                                # periodogram = periodogram_array,
                                re_matrices = re_matrices,
                                im_matrices = im_matrices,
                                prior_mean_Phi = prior_mean[1:d], 
                                diag_prior_var_Phi = diag(prior_var)[1:d],
                                prior_mean_gamma = prior_mean[(d+1):param_dim], 
                                diag_prior_var_gamma = diag(prior_var)[(d+1):param_dim],
                                # diag_prior_var_gamma = rep(0.1, 3),
                                transform = ifelse(transform == "arctanh", 1, 0)
                                )
  
  multi_sv_model_whittle <- cmdstan_model(
    stan_file_whittle,
    cpp_options = list(stan_threads = TRUE)
  )
  
  fit_stan_multi_sv_whittle <- multi_sv_model_whittle$sample(
    multi_sv_data_whittle,
    chains = n_chains,
    threads = parallel::detectCores(),
    refresh = 100,
    iter_warmup = burn_in,
    iter_sampling = n_post_samples
  )
  
  hmcw_results <- list(draws = fit_stan_multi_sv_whittle$draws(variables = c("Phi_mat", "Sigma_eta_mat")),
                       time = fit_stan_multi_sv_whittle$time)
  
  if (save_hmcw_results) {
    saveRDS(hmcw_results, hmcw_filepath)
  }
  
} else {
  hmcw_results <- readRDS(hmcw_filepath)
}

hmcw.post_samples_Phi <- hmcw_results$draws[,,1:(d^2)]
hmcw.post_samples_Sigma_eta <- hmcw_results$draws[,,(d^2+1):(2*d^2)]

# ## Plot posterior estimates
# indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
# hmc_indices <- diag(matrix(1:d^2, d, d, byrow = T))
# par(mfrow = c(d+1,d))
# 
# ### Posterior of diagonal entries of Phi  
# for (k in 1:d) {    
#   rvgaw.post_samples_phi <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[k,k]))
#   mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[k,k]))
#   
#   mcmcw.post_samples_phi <- mcmcw.post_samples_phi[-(1:burn_in)]
# 
#   ind <- paste0(k,k)
#   plot(density(rvgaw.post_samples_phi), col = "red", lty = 2, lwd = 2,
#        main = bquote(phi[.(ind)]), xlim = c(0.98, 1.0))
#   lines(density(mcmcw.post_samples_phi), col = "royalblue", lty = 2, lwd = 2)
#   lines(density(hmcw.post_samples_Phi[,,hmc_indices[k]]), col = "goldenrod", lwd = 2)
#   # lines(density(hmc.post_samples_Phi[,,hmc_indices[k]]), col = "deepskyblue", lwd = 2)
#   
#   # legend("topleft", legend = c("R-VGA Whittle", "MCMC Whittle", "HMC"), 
#   #        col = c("red", "royalblue", "deepskyblue"),
#   #        lty = c(2,2,1), cex = 0.7)
# }
# 
# ### Posterior of Sigma_eta
# # par(mfrow = c(1,d))
# hmc_indices <- c(matrix(1:(d^2), d, d)) #c(1,5,9)
# # for (k in 1:d) {
# for (k in 1:nrow(indices)) {
#   i <- indices[k, 1]
#   j <- indices[k, 2]
#   
#   rvgaw.post_samples_sigma_eta <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[i,j]))
#   mcmcw.post_samples_sigma_eta <- unlist(lapply(mcmcw.post_samples_Sigma_eta, function(x) x[i,j]))
#   mcmcw.post_samples_sigma_eta <- mcmcw.post_samples_sigma_eta[-(1:burn_in)]
#   ind <- paste0(i,j)
#   plot(density(rvgaw.post_samples_sigma_eta), col = "red", lty = 2, lwd = 2,
#     main = bquote(sigma_eta[.(ind)])) #, xlim = c(0, 2))
#   lines(density(mcmcw.post_samples_sigma_eta), col = "royalblue", lty = 2, lwd = 2)
#   lines(density(hmcw.post_samples_Sigma_eta[,,hmc_indices[k]]), col = "goldenrod", lwd = 2)
#   # lines(density(hmc.post_samples_Sigma_eta[,,hmc_indices[k]]), col = "deepskyblue", lwd = 2)
#   
#   # abline(v = Sigma_eta[i,j], lty = 2)
#   # legend("topright", legend = c("R-VGAW", "HMC"), col = c("red", "forestgreen"),
#   #        lty = c(2,2,1), cex = 0.3, y.intersp = 0.25)
# }
# 
# if (plot_trajectories) {
#   ## Parameter trajectories
#   par(mfrow = c(2,3))
#   trajectories <- list()
#   for (p in 1:param_dim) {
#     trajectory <- sapply(rvgaw_results$mu, function(e) e[p])
#     
#     if (p <= d) {
#       if (transform == "arctanh") {
#         trajectory <- tanh(trajectory)
#       } else {
#         trajectory <- 1/(1 + exp(-trajectory))
#       }
#     } else {
#       trajectory <- sqrt(exp(trajectory))
#     }
#     trajectories[[p]] <- trajectory
#     plot(trajectories[[p]], type = "l", xlab = "Iteration", ylab = "param", main = "")
#   }
# }
# 
# if (plot_trace) { # for mcmcw
#   par(mfrow = c(ceiling(d+d^2)/2, 2))
#   thinning_interval <- seq(1, iters, by = 1)
#   for (k in 1:d) {
#     mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[k,k]))
#     ## red = R-VGA Whittle, blue = MCMC Whittle, green = HMC
#     
#     if (transform == "arctanh") {
#       mcmcw.post_samples_phi <- tanh(mcmcw.post_samples_phi)
#     } else {
#       mcmcw.post_samples_phi <- 1/(1 + exp(-mcmcw.post_samples_phi))
#     }
#     
#     mcmcw.post_samples_phi <- mcmcw.post_samples_phi[-(1:burn_in)]
#     # mcmcw.post_samples_phi_thinned <- mcmcw.post_samples_phi[thinning_interval]                      
#     coda::traceplot(coda::as.mcmc(mcmcw.post_samples_phi), main = "phi")
#   }
#   
#   for (k in 1:nrow(indices)) {
#     i <- indices[k, 1]
#     j <- indices[k, 2]
#     
#     mcmcw.post_samples_sigma_eta <- unlist(lapply(mcmcw.post_samples_Sigma_eta, function(x) x[i,j]))
#     mcmcw.post_samples_sigma_eta <- mcmcw.post_samples_sigma_eta[-(1:burn_in)]
#     mcmcw.post_samples_sigma_eta <- sqrt(exp(mcmcw.post_samples_sigma_eta))
#     # mcmcw.post_samples_sigma_eta_thinned <- mcmcw.post_samples_sigma_eta[thinning_interval]
#     coda::traceplot(coda::as.mcmc(mcmcw.post_samples_sigma_eta), main = "sigma_eta")
#   }
#   
#   ## HMC trace plots
  # hmc.post_samples_phi11 <- as.vector(hmc.post_samples_Phi[,,1])
  # hmc.post_samples_phi22 <- as.vector(hmc.post_samples_Phi[,,4])
  # coda::traceplot(coda::as.mcmc(hmc.post_samples_phi11), ylab = "phi_11")
  # coda::traceplot(coda::as.mcmc(hmc.post_samples_phi22), ylab = "phi_22")
  
  # hmc.post_samples_sigma_eta11 <- as.vector(hmc.post_samples_Sigma_eta[,,1])
  # hmc.post_samples_sigma_eta21 <- as.vector(hmc.post_samples_Sigma_eta[,,2])
  # hmc.post_samples_sigma_eta22 <- as.vector(hmc.post_samples_Sigma_eta[,,4])
  
  # coda::traceplot(coda::as.mcmc(hmc.post_samples_sigma_eta11), ylab = "sigma_eta_11")
  # coda::traceplot(coda::as.mcmc(hmc.post_samples_sigma_eta21), ylab = "sigma_eta_21")
  # coda::traceplot(coda::as.mcmc(hmc.post_samples_sigma_eta22), ylab = "sigma_eta_22")
#   
# }
# 
# if (save_plots) {
#   plot_file <- paste0("svreal_posterior_", d, "d", "_", nobs, "obs", temper_info, reorder_info,
#                       "_", date, ".png")
#   filepath = paste0("./plots/", plot_file)
#   png(filepath, width = 600, height = 800)
#   
#   ## Plot posterior estimates
#   indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
#   hmc_indices <- diag(matrix(1:d^2, d, d, byrow = T))
#   par(mfrow = c(d+1,d))
#   
#   ### Posterior of diagonal entries of Phi  
#   for (k in 1:d) {    
#   rvgaw.post_samples_phi <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[k,k]))
#   mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[k,k]))
#   
#   ind <- paste0(k,k)
#   plot(density(rvgaw.post_samples_phi), col = "red", lty = 2, lwd = 2,
#        main = bquote(phi[.(ind)]), xlim = c(0.1, 0.99))
#   lines(density(mcmcw.post_samples_phi), col = "royalblue", lty = 2, lwd = 2)
#   lines(density(hmc.post_samples_Phi[,,hmc_indices[k]]), col = "deepskyblue")
#   lines(density(hmcw.post_samples_Phi[,,hmc_indices[k]]), col = "goldenrod")
#   
#   # legend("topleft", legend = c("R-VGA Whittle", "MCMC Whittle", "HMC"), 
#   #        col = c("red", "royalblue", "deepskyblue"),
#   #        lty = c(2,2,1), cex = 0.7)
# }
#   
#   ### Posterior of Sigma_eta
#   # par(mfrow = c(1,d))
#   hmc_indices <- c(matrix(1:(d^2), d, d)) #c(1,5,9)
#   # for (k in 1:d) {
#   for (k in 1:nrow(indices)) {
#     i <- indices[k, 1]
#     j <- indices[k, 2]
#     
#     rvgaw.post_samples_sigma_eta <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[i,j]))
#     mcmcw.post_samples_sigma_eta <- unlist(lapply(mcmcw.post_samples_Sigma_eta, function(x) x[i,j]))
#     
#     ind <- paste0(i,j)
#     plot(density(rvgaw.post_samples_sigma_eta), col = "red", lty = 2, lwd = 2,
#          main = bquote(sigma_eta[.(ind)]), xlim = c(0.1, 0.7))
#     lines(density(mcmcw.post_samples_sigma_eta), col = "royalblue", lty = 2, lwd = 2)
#     lines(density(hmc.post_samples_Sigma_eta[,,hmc_indices[k]]), lwd = 2, col = "deepskyblue")
#     lines(density(hmcw.post_samples_Sigma_eta[,,hmc_indices[k]]), lwd = 2, col = "goldenrod")
#     
#     # legend("topright", legend = c("R-VGAW", "HMC"), col = c("red", "forestgreen"),
#     #        lty = c(2,2,1), cex = 0.3, y.intersp = 0.25)
#   }
#   
#   dev.off()
#   
# }

## ggplot version
param_names <- c("phi11", "phi22", "sigma_eta11", "sigma_eta21", "sigma_eta22")
param_dim <- length(param_names)

ind_df <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d)) # (i,j) indices of elements in a dxd matrix

indmat <- matrix(1:d^2, d, d, byrow = T) # number matrix elements by row
phi_indices <- diag(indmat) # indices of diagonal elements of Phi
sigma_indices <- indmat[lower.tri(indmat, diag = T)] # lower triangular elements of Sigma_eta

rvgaw.post_samples <- matrix(NA, n_post_samples * n_chains, param_dim)
hmc.post_samples <- matrix(NA, n_post_samples * n_chains, param_dim)
hmcw.post_samples <- matrix(NA, n_post_samples * n_chains, param_dim)

# Arrange posterior samples of Phi in a matrix
for (k in 1:length(phi_indices)) {
  r <- phi_indices[k]
  i <- as.numeric(ind_df[r, ][1])
  j <- as.numeric(ind_df[r, ][2])
  rvgaw.post_samples[, k] <- sapply(rvgaw.post_samples_Phi, function(x) x[i,j])
  hmc.post_samples[, k] <- c(hmc.post_samples_Phi[,,r])
  hmcw.post_samples[, k] <- c(hmcw.post_samples_Phi[,,r])
}

# Arrange posterior samples of Sigma_eta in a matrix
for (k in 1:length(sigma_indices)) {
  r <- sigma_indices[k]
  i <- as.numeric(ind_df[r, ][1])
  j <- as.numeric(ind_df[r, ][2])
  rvgaw.post_samples[, k+d] <- sapply(rvgaw.post_samples_Sigma_eta, function(x) x[i,j])
  hmc.post_samples[, k+d] <- c(hmc.post_samples_Sigma_eta[,,r])
  hmcw.post_samples[, k+d] <- c(hmcw.post_samples_Sigma_eta[,,r])
}

throwaway <- 15000

rvgaw.df <- as.data.frame(rvgaw.post_samples[-(1:throwaway),])
hmc.df <- as.data.frame(hmc.post_samples[-(1:throwaway),])
hmcw.df <- as.data.frame(hmcw.post_samples[-(1:throwaway),])
names(rvgaw.df) <- param_names
names(hmc.df) <- param_names
names(hmcw.df) <- param_names

## ggplot version

plots <- list()

for (p in 1:param_dim) {
  
  plot <- ggplot(rvgaw.df, aes(x=.data[[param_names[p]]])) +
    # plot <- ggplot(exact_rvgal.df, aes(x=colnames(exact_rvgal.df)[p])) + 
    geom_density(col = "red", lwd = 1) +
    geom_density(data = hmcw.df, col = "goldenrod", lwd = 1) +
    geom_density(data = hmc.df, col = "deepskyblue", lwd = 1) +
    labs(x = vars) +
    theme_bw() +
    theme(axis.title = element_blank(), text = element_text(size = 24)) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 3))
  # theme(legend.position="bottom") + 
  # scale_color_manual(values = c('RVGA' = 'red', 'HMC' = 'blue'))
  
  plots[[p]] <- plot  
}

## Arrange bivariate plots in lower off-diagonals
n_lower_tri <- (param_dim^2 - param_dim)/2 # number of lower triangular elements

index_to_i_j_colwise_nodiag <- function(k, n) {
  kp <- n * (n - 1) / 2 - k
  p  <- floor((sqrt(1 + 8 * kp) - 1) / 2)
  i  <- n - (kp - p * (p + 1) / 2)
  j  <- n - 1 - p
  c(i, j)
}

cov_plots <- list()
for (ind in 1:n_lower_tri) {
  mat_ind <- index_to_i_j_colwise_nodiag(ind, param_dim)
  p <- mat_ind[1]
  q <- mat_ind[2]
  
  cov_plot <- ggplot(rvgaw.df, aes(x = .data[[param_names[q]]], y = .data[[param_names[p]]])) +
    stat_ellipse(col = "red", type = "norm", lwd = 1) +
    stat_ellipse(data = hmcw.df, col = "goldenrod", type = "norm", lwd = 1) +
    stat_ellipse(data = hmc.df, col = "deepskyblue", type = "norm", lwd = 1) +
    theme_bw() +
    theme(axis.title = element_blank(), text = element_text(size = 24)) +                               # Assign pretty axis ticks
    scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) 
  
  cov_plots[[ind]] <- cov_plot
}

m <- matrix(NA, param_dim, param_dim)
m[lower.tri(m, diag = F)] <- 1:n_lower_tri 
gr <- grid.arrange(grobs = cov_plots, layout_matrix = m)
gr2 <- gtable_add_cols(gr, unit(1, "null"), -1)
gr3 <- gtable_add_grob(gr2, grobs = lapply(plots, ggplotGrob), t = 1:param_dim, l = 1:param_dim)

# grid.draw(gr3)

# A list of text grobs - the labels
vars <- list(textGrob(bquote(Phi[11])), textGrob(bquote(Phi[22])),
             textGrob(bquote(Sigma[eta[11]])), textGrob(bquote(Sigma[eta[21]])),
             textGrob(bquote(Sigma[eta[21]])))
vars <- lapply(vars, editGrob, gp = gpar(col = "black", fontsize = 24))

# m <- matrix(1:param_dim, 1, param_dim, byrow = T)
# gr <- grid.arrange(grobs = plots, layout_matrix = m)
# gp <- gtable_add_rows(gr, unit(1.5, "lines"), -1) #0 adds on the top
# gtable_show_layout(gp)
# 
# gp <- gtable_add_grob(gp, vars[1:param_dim], t = 2, l = 1:3)

# So that there is space for the labels,
# add a row to the top of the gtable,
# and a column to the left of the gtable.
gp <- gtable_add_cols(gr3, unit(2, "lines"), 0)
gp <- gtable_add_rows(gp, unit(2, "lines"), -1) #0 adds on the top

# gtable_show_layout(gp)

# Add the label grobs.
# The labels on the left should be rotated; hence the edit.
# t and l refer to cells in the gtable layout.
# gtable_show_layout(gp) shows the layout.
gp <- gtable_add_grob(gp, lapply(vars[1:param_dim], editGrob, rot = 90), t = 1:param_dim, l = 1)
gp <- gtable_add_grob(gp, vars[1:param_dim], t = param_dim+1, l = 2:(param_dim+1))

grid.newpage()
grid.draw(gp)

if (save_plots) {
  plot_file <- paste0("test_multi_sv_real_posterior_", 
                      paste(currencies, collapse = "_"),
                      temper_info, reorder_info, block_info,
                      "_", transform, "_", date, ".png")
  filepath = paste0("./plots/", plot_file)
  png(filepath, width = 1200, height = 900)
  grid.draw(gp)
  dev.off()
}

## Timings
# rvgaw.time <- rvgaw_results$time_elapsed[3]
# hmcw.time <- sum(hmcw_results$time()$chains$total)
# hmc.time <- sum(hmc_results$time()$chains$total)
# print(data.frame(method = c("R-VGA", "HMCW", "HMC"),
#                  time = c(rvgaw.time, hmcw.time, hmc.time)))