## Bivariate SV model
rm(list = ls())
setwd("~/R-VGA-Whittle/04_Multi_SV_Sigma/")

## Flags
# date <- "20230920" #"20230918" has 5D, "20230920" has 3D
# date <- "20230918"
date <- "20240613" #"20240227"
regenerate_data <- F
save_data <- F
use_cholesky <- T # use lower Cholesky factor to parameterise Sigma_eta
transform <- "arctanh"
prior_type <- "prior1"
use_heaps_mapping <- F
plot_likelihood_surface <- F
plot_prior_samples <- T
plot_trace <- F
plot_trajectories <- F
save_plots <- F

rerun_rvgaw <- F
rerun_mcmcw <- F
rerun_hmc <- F
rerun_hmcw <- F

save_rvgaw_results <- F
save_mcmcw_results <- F
save_hmc_results <- F
save_hmcw_results <- F

## R-VGAW flags
use_tempering <- T #T
temper_first <- T
reorder <- 0 #"decreasing"
reorder_seed <- 2024
# decreasing <- T
use_median <- F
# nblocks <- 100

## HMC flags
n_post_samples <- 10000
burn_in <- 5000
n_chains <- 2

library(mvtnorm)
library(astsa)
# library(stcos)
library(coda)
library(Matrix)
# library("expm")
reticulate::use_condaenv("myenv", required = TRUE)
library(tensorflow)
tfp <- import("tensorflow_probability")
tfd <- tfp$distributions
library(keras)
library(cmdstanr)
library(ggplot2)
library(grid)
library(gridExtra)
library(gtable)

# source("./source/run_rvgaw_multi_sv.R")
source("./source/run_rvgaw_multi_sv_block.R")
source("./source/run_mcmc_multi_sv.R")
source("./source/compute_whittle_likelihood_multi_sv.R")
source("./source/compute_periodogram.R")
source("./source/compute_periodogram_uni.R")
# source("./source/run_mcmc_sv.R")
# source("./source/compute_whittle_likelihood_sv.R")
source("./source/construct_prior2.R")
# source("./source/map_functions.R")
source("./source/construct_Sigma.R")
# source("./archived/compute_partial_whittle_likelihood.R")
# source("./source/compute_grad_hessian.R")
source("./source/compute_grad_hessian_block.R")
source("./source/find_cutoff_freq.R")

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

dataset <- "" #"hmc_est" 

Tfin <- 5000
d <- 2L
if (regenerate_data) {
#  if (dataset == "5") {
    Phi <- diag(c(0.99, 0.98))
    Sigma_eta <- matrix(c(0.02, 0.005, 0.005, 0.01), 2, 2)
  # } else if (dataset == "hmc_est") {
  #   Phi <- diag(c(0.96, 0.97))
  #   Sigma_eta <- matrix(c(0.18, 0.11, 0.11, 0.125), 2, 2)
  # } else { # generate a random Phi matrix
  #   Phi <- matrix(c(0.7, 0, 0, 0.8), 2, 2)
  #   Sigma_eta <- matrix(c(0.4, 0.05, 0.05, 0.2), 2, 2)
  # } 
  
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
    saveRDS(multi_sv_data, file = paste0("./data/multi_sv_data_", d, "d_Tfin", Tfin, "_", date, ".rds"))
  }
} else {
  multi_sv_data <- readRDS(file = paste0("./data/multi_sv_data_", d, "d_Tfin", Tfin, "_20240227.rds"))
}

X <- multi_sv_data$X
Y <- multi_sv_data$Y
Phi <- multi_sv_data$Phi
Sigma_eta <- multi_sv_data$Sigma_eta
Sigma_eps <- multi_sv_data$Sigma_eps

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
# browser()
############################ Plot periodogram ##################################
# Z <- log(Y^2) - colMeans(log(Y^2))
# pgram_out <- compute_periodogram(Z)
# freq <- pgram_out$freq
# I <- pgram_out$periodogram

# I11 <- sapply(1:length(freq), function(i) I[,,i][1,1])
# I22 <- sapply(1:length(freq), function(i) I[,,i][2,2])
# I21 <- sapply(1:length(freq), function(i) I[,,i][2,1])

# coherence <- Mod(I21)^2 / (I11 * I22)

# browser()
# coherence <- sapply(1:length(freq), function(i) Mod(I[,,i][2,1])^2 / (I[,,i][1,1] * I[,,i][2,2]))

# test1 <- compute_periodogram_uni(Y[, 1])$periodogram
# test2 <- compute_periodogram_uni(Y[, 2])$periodogram

# df1 <- data.frame(freq = freq, periodogram = test1)
# df2 <- data.frame(freq = freq, periodogram = test2)

# df1 %>% ggplot() + geom_line(aes(x = freq, y = periodogram)) + 
#   ggtitle("Periodogram for series 1")

# df2 %>% ggplot() + geom_line(aes(x = freq, y = periodogram)) + 
#   ggtitle("Periodogram for series 2")

# library(stats)
# # Compute the cross-spectral density
# csd_result <- spectrum(Y)
# browser()
############################## Inference #######################################

## Result directory
result_directory <- paste0("./results/", d, "d/")

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
  prior_samples <- rmvnorm(10000, prior_mean, prior_var)
  
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
  # construct_Sigma_eta <- function(theta, d) {
  #   p <- length(theta)
  #   L <- diag(exp(theta[(d+1):(p-1)]))
  #   L[2,1] <- theta[p]
  #   Sigma_eta <- L %*% t(L)
  #   # Sigma_eta <- L
  #   return(Sigma_eta)
  # }
  
  Sigma_eta_prior_samples <- lapply(prior_samples_list, construct_Sigma_eta, d = d)
  # sigma11 <- lapply(Sigma_eta_prior_samples, function(x) x[1,1])
  # sorted <- sort(unlist(sigma11))
  # sorted[250]
  
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
  
  png("./plots/test_prior.png", width = 1800, height = 1800)
  par(mfrow = c(6,6))
  for (i in 1:6) {
    for (j in 1:6) {
      par("mar"=c(4, 4, 2, 2))
      plot(VAR1_prior_samples[[i]], VAR1_prior_samples[[j]], 
           xlab = param_names[i], ylab = param_names[j])
      points(true_params[i], true_params[j], col = "red", pch = 20)
    }
  }
  dev.off()

  quantiles <- lapply(VAR1_prior_samples, quantile, probs = c(0.025, 0.975))

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
  # browser()
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

################################
##    R-VGAW implementation   ##
################################

blocksize <- 100

nsegs <- 25
power_prop <- 1/2

c1 <- find_cutoff_freq(Y[, 1], nsegs = nsegs, power_prop = power_prop)$cutoff_ind
c2 <- find_cutoff_freq(Y[, 2], nsegs = nsegs, power_prop = power_prop)$cutoff_ind
n_indiv <- max(c1, c2)

# browser()

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
# a_vals <- 1

################ R-VGA starts here #################
print("Starting R-VGAL with Whittle likelihood...")

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_Tfin", Tfin, 
                         temper_info, reorder_info, block_info, "_", date, 
                         prior_type, ".rds")

if (rerun_rvgaw) {
  rvgaw_results <- run_rvgaw_multi_sv(data = Y, 
                                      n_post_samples = n_post_samples * n_chains,
                                      prior_mean = prior_mean, 
                                      prior_var = prior_var, S = S,
                                      use_cholesky = use_cholesky,
                                      transform = transform,
                                      use_tempering = use_tempering, 
                                      temper_first = temper_first,
                                      temper_schedule = temper_schedule, 
                                      reorder = reorder, 
                                      reorder_seed = reorder_seed,
                                      # decreasing = decreasing,
                                      # nblocks = nblocks,
                                      blocksize = blocksize,
                                      n_indiv = n_indiv)
                                      # use_median = use_median)
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
# print("Starting MCMC with Whittle likelihood...")

# mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_results_Tfin", Tfin, 
#                          "_", date, prior_type, ".rds")

# # n_post_samples <- 50000
# # burn_in <- 50000
# iters <- (n_post_samples + burn_in) * n_chains

# if (rerun_mcmcw) {
#   mcmcw_results <- run_mcmc_multi_sv(data = Y, iters = iters, burn_in = burn_in, 
#                                      prior_mean = prior_mean, prior_var = prior_var,
#                                      adapt_proposal = T, use_whittle_likelihood = T,
#                                      use_cholesky = use_cholesky, transform = transform)
#   if (save_mcmcw_results) {
#     saveRDS(mcmcw_results, mcmcw_filepath)
#   }
# } else {
#   mcmcw_results <- readRDS(mcmcw_filepath)
# }

# ## Extract samples
# mcmcw.post_samples_Phi_og <- lapply(mcmcw_results$post_samples, function(x) x$Phi) #post_samples_theta[, 1]
# mcmcw.post_samples_Sigma_eta_og <- lapply(mcmcw_results$post_samples, function(x) x$Sigma_eta) #post_samples_theta[, 2]
# mcmcw.post_samples_Phi <- mcmcw.post_samples_Phi_og[-(1:burn_in)]
# mcmcw.post_samples_Sigma_eta <- mcmcw.post_samples_Sigma_eta_og[-(1:burn_in)]

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
                       "_", date, prior_type, ".rds")

if (rerun_hmc) {
  
  # n_post_samples <- 10000
  # burn_in <- 5000
  # stan.iters <- n_post_samples + burn_in
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
print("Starting HMC with Whittle likelihood...")

hmcw_filepath <- paste0(result_directory, "hmcw_results_Tfin", Tfin, 
                       "_", date, prior_type, ".rds")

if (rerun_hmcw) {
  
  # n_post_samples <- 10000
  # burn_in <- 5000
  # stan.iters <- n_post_samples + burn_in
  
  stan_file_whittle <- "./source/stan_multi_sv_whittle.stan"
  
  # ## Calculation of Whittle likelihood
  # ## Fourier frequencies
  # k <- seq(-ceiling(Tfin/2)+1, floor(Tfin/2), 1)
  # k_in_likelihood <- k [k >= 1 & k <= floor((Tfin-1)/2)]
  # freq <- 2 * pi * k_in_likelihood / Tfin
  
  # # ## astsa package
  Z <- log(Y^2) - colMeans(log(Y^2))
  # fft_out <- mvspec(Z, detrend = F, plot = F)
  # I <- fft_out$fxx

  pgram_out <- compute_periodogram(Z)
  freq <- pgram_out$freq
  I <- pgram_out$periodogram
  
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
                                transform = ifelse(transform == "arctanh", 1, 0)
                                # truePhi = Phi,
                                # trueSigma = Sigma_eta
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
                       time = fit_stan_multi_sv_whittle$time())
  
  if (save_hmcw_results) {
    saveRDS(hmcw_results, hmcw_filepath)
  }
  
} else {
  hmcw_results <- readRDS(hmcw_filepath)
}

hmcw.post_samples_Phi <- hmcw_results$draws[,,1:(d^2)]
hmcw.post_samples_Sigma_eta <- hmcw_results$draws[,,(d^2+1):(2*d^2)]

# ## Posterior density comparisons

# if (plot_trace) { # for mcmcw
  
#   IF <- c()
#   ESS <- c()
#   for (k in 1:d) {
#     mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[k,k]))
#     ## red = R-VGA Whittle, blue = MCMC Whittle, green = HMC
    
#     # mcmcw.post_samples_phi <- mcmcw.post_samples_phi[-(1:burn_in)]
#     mcmcw.post_samples_phi <- as.mcmc(mcmcw.post_samples_phi)

#     ESS[k] <- coda::effectiveSize(mcmcw.post_samples_phi)
    
#     #Compute Inefficiency factor
#     IF[k] <- length(mcmcw.post_samples_phi)/ESS[k]
    
#     # mcmcw.post_samples_phi <- mcmcw.post_samples_phi[thinning_interval]
#     coda::traceplot(coda::as.mcmc(mcmcw.post_samples_phi))
#   }
  
#   for (k in 1:nrow(indices)) {
#     i <- indices[k, 1]
#     j <- indices[k, 2]
    
#     mcmcw.post_samples_sigma_eta <- unlist(lapply(mcmcw.post_samples_Sigma_eta, function(x) x[i,j]))
#     # mcmcw.post_samples_sigma_eta <- mcmcw.post_samples_sigma_eta[-(1:burn_in)]
#     mcmcw.post_samples_sigma_eta <- as.mcmc(mcmcw.post_samples_sigma_eta)
    
#     ESS[k] <- coda::effectiveSize(mcmcw.post_samples_sigma_eta)
    
#     # Compute Inefficiency factor
#     IF[k] <- length(mcmcw.post_samples_sigma_eta)/ESS[k]
    
#     # mcmcw.post_samples_sigma_eta_thinned <- mcmcw.post_samples_sigma_eta[thinning_interval]
#     coda::traceplot(coda::as.mcmc(mcmcw.post_samples_sigma_eta))
#   }
  
#   # ## Thinning
#   # thinning_interval <- seq(1, iters-burn_in, by = ceiling(max(IF)))
#   # 
#   # for (k in 1:d) {
#   #   mcmcw.post_samples_phi_thinned <- mcmcw.post_samples_phi[thinning_interval]                      
#   #   coda::traceplot(coda::as.mcmc(mcmcw.post_samples_phi_thinned))
#   # }
    
# }

# if (plot_trajectories) {
#   ## Parameter trajectories
#   par(mfrow = c(2,3))
#   trajectories <- list()
#   for (p in 1:param_dim) {
#     trajectories[[p]] <- sapply(rvgaw_results$mu, function(e) e[p])
#     plot(trajectories[[p]], type = "l", xlab = "Iteration", ylab = "param", main = "")
#   }
# }

# ## Trajectories
# if (plot_trajectories) {
#   mu_phi <- lapply(rvgaw_results$mu, function(x) x[1:2])
#   mu_sigma_eta <- lapply(rvgaw_results$mu, function(x) x[3:5])

#   ## Back-transform to original scale
#   if (transform == "arctanh") {
#     mu_phi <- tanh(mu_phi)
#   } else { # logit transform
#     mu_phi <- exp(mu_phi) / (1 + exp(mu_phi))
#   }
#   mu_sigma_eta <- sqrt(exp(mu_sigma_eta[3:4]))

#   param_names <- c("phi11", "phi22", "sigma[eta[11]]", "sigma[eta[22]]", "sigma[eta[21]]")

#   true_df <- data.frame(param = param_names, 
#                         value = c(diag(Phi), Sigma_eta[lower.tri(Sigma_eta, diag = T)]))

#   trajectory_df <- data.frame(phi = mu_phi, sigma_eta = mu_sigma_eta)
#   names(trajectory_df) <- c("phi", "sigma[eta]")
#   trajectory_df$iter <- 1:nrow(trajectory_df)

#   trajectory_df_long <- trajectory_df %>% pivot_longer(cols = !iter, 
#                                                       names_to = "param", values_to = "value")
#   trajectory_plot <- trajectory_df_long %>% ggplot() + 
#     geom_line(aes(x = iter, y = value), linewidth = 1) +
#     facet_wrap(~param, scales = "free", labeller = label_parsed) +
#     geom_hline(data = true_df, aes(yintercept = value), linetype = "dashed", linewidth = 1.5) +
#     theme_bw() + theme(text = element_text(size = 28)) + 
#     xlab("Iterations") + ylab("Value")


#   png(paste0("plots/trajectories_sv_sim", block_info, ".png"), width = 1000, height = 500)
#   # par(mfrow = c(1, 2))
#   # if (transform == "arctanh") {
#   #   plot(tanh(mu_phi), type = "l", main = "Trajectory of phi")
#   # } else {
#   #   plot(1 / (1 + exp(-mu_phi)), type = "l", main = "Trajectory of phi")
#   # }
#   # abline(h = phi, lty = 2)
#   # plot(sqrt(exp(mu_eta)), type = "l", main = "Trajectory of sigma_eta")
#   # abline(h = sigma_eta, lty = 2)
#     print(trajectory_plot)                                                      

#   dev.off()
# }


## ggplot version
param_names <- c("phi11", "phi22", "sigma_eta11", "sigma_eta21", "sigma_eta22")
param_dim <- length(param_names)
param_values <- c(diag(Phi), Sigma_eta[lower.tri(Sigma_eta, diag = T)])

ind_df <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d)) # (i,j) indices of elements in a dxd matrix

indmat <- matrix(1:d^2, d, d, byrow = T) # number matrix elements by row
phi_indices <- diag(indmat) # indices of diagonal elements of Phi
sigma_indices <- indmat[lower.tri(indmat, diag = T)] # lower triangular elements of Sigma_eta

thin_interval <- 50
rvgaw.post_samples <- matrix(NA, length(rvgaw.post_samples_Phi), param_dim)
hmc.post_samples <- matrix(NA, n_post_samples*n_chains/thin_interval, param_dim)
hmcw.post_samples <- matrix(NA, n_post_samples*n_chains, param_dim)

hmc.ESS <- c()
hmc.IF <- c()
hmc.acf <- list()

hmcw.ESS <- c()
hmcw.IF <- c()
hmcw.acf <- list()

# Arrange posterior samples of Phi in a matrix
for (k in 1:length(phi_indices)) {
  r <- phi_indices[k]
  i <- as.numeric(ind_df[r, ][1])
  j <- as.numeric(ind_df[r, ][2])
  rvgaw.post_samples[, k] <- sapply(rvgaw.post_samples_Phi, function(x) x[i,j])

  hmc_samples <- mcmc(c(hmc.post_samples_Phi[,,r]))
  hmcw_samples <- mcmc(c(hmcw.post_samples_Phi[,,r]))

  ## ACF
  hmc.acf[[k]] <- autocorr(hmc_samples, lags = c(0, 1, 5, 10, 20, 50, 100), relative=F)
  hmcw.acf[[k]] <- autocorr(hmcw_samples, lags = c(0, 1, 5, 10, 20, 50, 100), relative=F)

  ## Effective Sample Size
  hmc.ESS[k] <- coda::effectiveSize(hmc_samples)
  hmcw.ESS[k] <- coda::effectiveSize(hmcw_samples)
    
  # Compute Inefficiency factor
  hmc.IF[k] <- length(hmc_samples)/hmc.ESS[k]
  hmcw.IF[k] <- length(hmcw_samples)/hmcw.ESS[k]

  ## Thin samples
  
  hmc.post_samples[, k] <- as.vector(window(hmc_samples, thin = thin_interval))
  hmcw.post_samples[, k] <- as.vector(window(hmcw_samples, thin = 1))
}

# Arrange posterior samples of Sigma_eta in a matrix
for (k in 1:length(sigma_indices)) {
  r <- sigma_indices[k]
  i <- as.numeric(ind_df[r, ][1])
  j <- as.numeric(ind_df[r, ][2])
  rvgaw.post_samples[, k+d] <- sapply(rvgaw.post_samples_Sigma_eta, function(x) x[i,j])

  hmc_samples <- mcmc(c(hmc.post_samples_Sigma_eta[,,r]))
  hmcw_samples <- mcmc(c(hmcw.post_samples_Sigma_eta[,,r]))

  ## ACF
  hmc.acf[[k+d]] <- autocorr(hmc_samples, lags = c(0, 1, 5, 10, 20, 50), relative=F)
  hmcw.acf[[k+d]] <- autocorr(hmcw_samples, lags = c(0, 1, 5, 10, 20, 50), relative=F)

  ## Effective Sample Size
  hmc.ESS[k+d] <- coda::effectiveSize(hmc_samples)
  hmcw.ESS[k+d] <- coda::effectiveSize(hmcw_samples)
    
  ## Compute Inefficiency factor
  hmc.IF[k+d] <- length(hmc_samples)/hmc.ESS[k+d^2]
  hmcw.IF[k+d] <- length(hmcw_samples)/hmcw.ESS[k+d^2]

  ## Thin samples
  hmc.post_samples[, k+d] <- as.vector(window(hmc_samples, thin = thin_interval))
  hmcw.post_samples[, k+d] <- as.vector(window(hmcw_samples, thin = 1))
}

rvgaw.df <- as.data.frame(rvgaw.post_samples)
hmc.df <- as.data.frame(hmc.post_samples)
hmcw.df <- as.data.frame(hmcw.post_samples)
names(rvgaw.df) <- param_names
names(hmc.df) <- param_names
names(hmcw.df) <- param_names


plots <- list()

xlims <- list(c(0.97, 1), c(0.95, 1), c(0, 0.04), c(0, 0.01), c(0, 0.02))

for (p in 1:param_dim) {
  
  true_vals.df <- data.frame(name = param_names[p], val = param_values[p])


  plot <- ggplot(rvgaw.df, aes(x=.data[[param_names[p]]])) +
    # plot <- ggplot(exact_rvgal.df, aes(x=colnames(exact_rvgal.df)[p])) + 
    geom_density(col = "red", lwd = 1) +
    geom_density(data = hmcw.df, col = "goldenrod", lwd = 1) +
    geom_density(data = hmc.df, col = "deepskyblue", lwd = 1) +
    geom_vline(data = true_vals.df, aes(xintercept=val),
               color="black", linetype="dashed", linewidth=1) +
    labs(x = vars) +
    xlim(x = xlims[[p]]) +
    theme_bw() +
    theme(axis.title = element_blank(), text = element_text(size = 24))
    # scale_x_continuous(breaks = scales::pretty_breaks(n = 3))
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
  
  param_df <- data.frame(x = param_values[q], y = param_values[p])

  cov_plot <- ggplot(rvgaw.df, aes(x = .data[[param_names[q]]], y = .data[[param_names[p]]])) +
    stat_ellipse(col = "red", type = "norm", lwd = 1) +
    stat_ellipse(data = hmcw.df, col = "goldenrod", type = "norm", lwd = 1) +
    stat_ellipse(data = hmc.df, col = "deepskyblue", type = "norm", lwd = 1) +
    geom_point(data = param_df, aes(x = x, y = y),
               shape = 4, color = "black", size = 4) +
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
gp <- gtable_add_cols(gr3, unit(1.5, "lines"), 0)
gp <- gtable_add_rows(gp, unit(1.5, "lines"), -1) #0 adds on the top

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
  # plot_file <- paste0("multi_sv_sim_posterior", "_", Tfin, temper_info, reorder_info, block_info,
  #                     "_", transform, "_thinned_", date, ".png")
  plot_file <- paste0("test_multi_sv_sim_posterior", "_", Tfin, temper_info, reorder_info, block_info,
                      "_", transform, "_thinned", ".png")
  filepath = paste0("./plots/", plot_file)
  png(filepath, width = 1200, height = 900)
  grid.draw(gp)
  dev.off()
}

## Timings
rvgaw.time <- rvgaw_results$time_elapsed[3]
hmcw.time <- sum(hmcw_results$time$chains$total)
hmc.time <- sum(hmc_results$time()$chains$total)
print(data.frame(method = c("R-VGA", "HMCW", "HMC"),
                 time = c(rvgaw.time, hmcw.time, hmc.time)))
