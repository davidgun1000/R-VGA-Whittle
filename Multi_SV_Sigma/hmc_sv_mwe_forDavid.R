## R-VGAW on the SV model

## MCMC on multivariate SV model
rm(list = ls())
setwd("~/R-VGA-Whittle/Multi_SV_Sigma/")

## Flags
# date <- "20230920" #"20230918" has 5D, "20230920" has 3D
date <- "20230918"
regenerate_data <- F
save_data <- F
use_cholesky <- T # use lower Cholesky factor to parameterise Sigma_eta
transform <- "logit"
prior_type <- "prior1"
plot_likelihood_surface <- F
plot_prior_samples <- F
# plot_trace <- F
# plot_trajectories <- F
save_plots <- F

rerun_hmc <- T
save_hmc_results <- F

library("mvtnorm")
library("astsa")
library("stcos")
library("coda")
library(Matrix)
library("cmdstanr")

# source("./source/run_rvgaw_multi_sv.R")
# source("./source/run_mcmc_multi_sv.R")
# source("./source/compute_whittle_likelihood_multi_sv.R")
# source("./source/run_mcmc_sv.R")
# source("./source/compute_whittle_likelihood_sv.R")
source("./source/construct_prior2.R")
# source("./source/map_functions.R")
# source("./source/construct_Sigma.R")
# source("./archived/compute_partial_whittle_likelihood.R")
# source("./source/compute_grad_hessian.R")

#######################
##   Generate data   ##
#######################

dataset <- "5" 

Tfin <- 10000
d <- 2L

## Result directory
result_directory <- paste0("./results/", d, "d/")

if (regenerate_data) {
  
  Phi <- diag(0.1*c(9:(9-d+1)))
  Sigma_eps <- diag(d)
  
  if (dataset == "0") {
    Sigma_eta <- diag(0.1*(1:d))
    
  } else if (dataset == "5") {
    nlower <- d*(d-1)/2
    diags <- 0.1*(1:d)
    lowers <- 0.05*(1:nlower)
    Sigma_eta <- diag(diags)
    Sigma_eta[lower.tri(Sigma_eta)] <- lowers
    Sigma_eta[upper.tri(Sigma_eta)] <- t(Sigma_eta)[upper.tri(Sigma_eta)]
    
  }  else { # generate a random Phi matrix
    A <- matrix(rnorm(d^2), d, d)
    Phi <- backward_map(A, Sigma_eta)
    Phi <- round(Phi, digits = 1)
    print(Phi)
  }
  
  x1 <- rep(0, d)
  X <- matrix(NA, nrow = Tfin+1, ncol = d) # x_0:T
  X[1, ] <- t(x1)
  Y <- matrix(NA, nrow = Tfin, ncol = d) # y_1:T
  # Y[, 1] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
  set.seed(2023)
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
# 
for (k in 1:d) {
  plot(Y[, k], type = "l")
}
# par(mfrow = c(1,2))
# hist(Y[1,])
# hist(Y[2,])
############################## Inference #######################################

## Construct initial distribution/prior
prior <- construct_prior(data = Y, prior_type = prior_type, use_cholesky = use_cholesky)
prior_mean <- prior$prior_mean
prior_var <- prior$prior_var

d <- as.integer(ncol(Y))
Tfin <- as.integer(nrow(Y))
param_dim <- length(prior_mean)

if (plot_prior_samples) {
  ## Sample from the priors here
  prior_samples <- rmvnorm(1000, prior_mean, prior_var)
  
  d <- nrow(Phi)
  indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
  
  ### the first 4 elements will be used to construct Phi
  prior_samples_list <- lapply(seq_len(nrow(prior_samples)), function(i) prior_samples[i,])
  
  Phi_prior_samples <- lapply(prior_samples_list, function(x) diag(tanh(x[1:d])))
  
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
  
  
  VAR1_prior_samples <- list()
  
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

if (prior_type == "minnesota") { # renaming for file saving purposes
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


########################
###       STAN       ###
########################
print("Starting HMC...")

hmc_filepath <- paste0(result_directory, "hmc_results_Tfin", Tfin, 
                       "_", date, "_", dataset, prior_type, ".rds")


if (rerun_hmc) {
  n_post_samples <- 1000#0
  burn_in <- 1000
  stan.iters <- n_post_samples + burn_in
  d <- as.integer(ncol(Y))
  
  # use_chol <- 0
  # if (use_cholesky) {
  #   use_chol <- 1
  # }
  
  stan_file <- "./source/stan_multi_sv.stan"
  multi_sv_data <- list(d = d, Tfin = Tfin, Y = Y,
                        prior_mean_Phi = prior_mean[1:d], 
                        diag_prior_var_Phi = diag(prior_var)[1:d],
                        prior_mean_gamma = prior_mean[(d+1):param_dim], 
                        diag_prior_var_gamma = diag(prior_var)[(d+1):param_dim],
                        transform = ifelse(transform == "arctanh", 1, 0)
  )
  
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

## Plot posterior estimates
indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
hmc_indices <- diag(matrix(1:d^2, d, d, byrow = T))
par(mfrow = c(d+1,d))

plot_margin <- c(-0.1, 0.1)

### Posterior of diagonal entries of Phi  
for (k in 1:d) {    
  # rvgaw.post_samples_phi <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[k,k]))
  # mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[k,k]))
  # hmc.post_samples_phi <- unlist(lapply(hmc.post_samples_Phi, function(x) x[k,k]))
  
  ind <- paste0(k,k)
  plot(density(hmc.post_samples_Phi[,,hmc_indices[k]]), col = "royalblue", lty = 3, lwd = 2, 
       main = bquote(phi[.(ind)])) #, xlim = Phi[k,k] + plot_margin)
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
  
  # rvgaw.post_samples_sigma_eta <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[i,j]))
  # mcmcw.post_samples_sigma_eta <- unlist(lapply(mcmcw.post_samples_Sigma_eta, function(x) x[i,j]))
  
  ind <- paste0(i,j)
  plot(density(hmc.post_samples_Sigma_eta[,,hmc_indices[k]]), col = "royalblue", lty = 3, lwd = 2, 
       main = bquote(sigma_eta[.(ind)]), xlim = Sigma_eta[i,j] + plot_margin)
  # lines(density(mcmcw.post_samples_sigma_eta), col = "royalblue", lty = 3, lwd = 2)
  # lines(density(hmc.post_samples_Sigma_eta[,,hmc_indices[k]]), lty = 1, col = "skyblue", lwd = 2)
  abline(v = Sigma_eta[i,j], lty = 2, lwd = 2)
  # legend("topright", legend = c("R-VGAW", "HMC"), col = c("red", "forestgreen"),
  #        lty = c(2,2,1), cex = 0.3, y.intersp = 0.25)
}

# if (plot_trace) { # for mcmcw
#   
#   thinning_interval <- seq(1, iters, by = 1)
#   for (k in 1:d) {
#     mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[k,k]))
#     ## red = R-VGA Whittle, blue = MCMC Whittle, green = HMC
#     
#     mcmcw.post_samples_phi <- mcmcw.post_samples_phi[-(1:burn_in)]
#     mcmcw.post_samples_phi_thinned <- mcmcw.post_samples_phi[thinning_interval]                      
#     coda::traceplot(coda::as.mcmc(mcmcw.post_samples_phi_thinned))
#   }
#   
#   for (k in 1:nrow(indices)) {
#     i <- indices[k, 1]
#     j <- indices[k, 2]
#     
#     mcmcw.post_samples_sigma_eta <- unlist(lapply(mcmcw.post_samples_Sigma_eta, function(x) x[i,j]))
#     mcmcw.post_samples_sigma_eta <- mcmcw.post_samples_sigma_eta[-(1:burn_in)]
#     mcmcw.post_samples_sigma_eta_thinned <- mcmcw.post_samples_sigma_eta[thinning_interval]
#     coda::traceplot(coda::as.mcmc(mcmcw.post_samples_sigma_eta_thinned))
#   }
#   
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

# if (save_plots) {
#   plot_file <- paste0("sv_posterior_", d, "d", temper_info, reorder_info,
#                       "_", date, ".png")
#   filepath = paste0("./plots/", plot_file)
#   png(filepath, width = 1000, height = 1000)
#   
#   ## Plot posterior estimates
#   indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
#   hmc_indices <- diag(matrix(1:d^2, d, d, byrow = T))
#   par(mfrow = c(d+1,d))
#   
#   plot_margin <- c(-0.1, 0.1)
#   
#   ### Posterior of diagonal entries of Phi  
#   for (k in 1:d) {    
#     rvgaw.post_samples_phi <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[k,k]))
#     mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[k,k]))
#     # hmc.post_samples_phi <- unlist(lapply(hmc.post_samples_Phi, function(x) x[k,k]))
#     
#     ind <- paste0(k,k)
#     plot(density(rvgaw.post_samples_phi), col = "red", lty = 3, lwd = 2, 
#          main = bquote(phi[.(ind)]), xlim = Phi[k,k] + plot_margin)
#     lines(density(mcmcw.post_samples_phi), col = "royalblue", lty = 3, lwd = 2)
#     # lines(density(hmc.post_samples_phi), col = "skyblue")
#     
#     lines(density(hmc.post_samples_Phi[,,hmc_indices[k]]), col = "skyblue", lty = 1, lwd = 2)
#     abline(v = Phi[k,k], lty = 2, lwd = 2)
#     # legend("topright", legend = c("R-VGAW", "HMC"), col = c("red", "forestgreen"),
#     #        lty = c(2,2,1), cex = 0.7)
#   }
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
#     plot(density(rvgaw.post_samples_sigma_eta), col = "red", lty = 3, lwd = 2, 
#          main = bquote(sigma_eta[.(ind)]), xlim = Sigma_eta[i,j] + plot_margin)
#     lines(density(mcmcw.post_samples_sigma_eta), col = "royalblue", lty = 3, lwd = 2)
#     lines(density(hmc.post_samples_Sigma_eta[,,hmc_indices[k]]), lty = 1, col = "skyblue", lwd = 2)
#     abline(v = Sigma_eta[i,j], lty = 2, lwd = 2)
#     # legend("topright", legend = c("R-VGAW", "HMC"), col = c("red", "forestgreen"),
#     #        lty = c(2,2,1), cex = 0.3, y.intersp = 0.25)
#   }
#   
#   dev.off()
# }
