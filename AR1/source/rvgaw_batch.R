## R-VGA with reprocessing of observations at every step

setwd("~/R-VGA-Whittle/AR1")

## R-VGA with Whittle likelihood (R-VGAW)?
rm(list = ls())

# library(stats)
# library(LSTS)
library(Matrix)
library(coda)
library(ggplot2)
library(tidyr)
library(mvtnorm)
source("./source/calculate_likelihood.R")
source("./source/run_rvgaw_ar1.R")
# source("./source/run_rvgae_ar1_archived.R")
# source("./source/run_mcmc_ar1.R")
# source("./source/run_vb_ar1.R")

result_directory <- "./results/"

## Flags
date <- "20230417" # 20240410 has phi = 0.9, 20230417 has phi = 0.7
regenerate_data <- T
save_data <- F

## R-VGA flags
rerun_rvgaw <- T
save_rvgaw_results <- F
use_tempering <- F
reorder_freq <- F
reorder_seed <- 2024
decreasing <- F
transform <- "arctanh"

## Model parameters 
phi <- 0.99
sigma_e <- 0.5
n <- 1000 # time series length

if (use_tempering) {
  n_temper <- 0.1 * n #floor(n/2) #10
  temper_schedule <- rep(1/10, 10)
  temper_info <- paste0("_temper", n_temper)
} else {
  n_temper <- 0
  temper_schedule <- NULL
  temper_info <- ""
}

if (reorder_freq) {
  reorder_info <- "_reordered"
} else {
  reorder_info <- ""
}

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

if (regenerate_data) {
  ## Generate AR(1) series
  set.seed(2023)
  x0 <- rnorm(1, 0, 1)
  x <- c()
  x[1] <- x0
  
  for (t in 2:n) {
    x[t] <- phi * x[t-1] + rnorm(1, 0, sigma_e) 
  }
  rvgaw_data <- list(x = x, phi = phi, sigma_e = sigma_e)
  
  if (save_data) {
    saveRDS(rvgaw_data, file = paste0("./data/ar1_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  }
  
} else {
  rvgaw_data <- readRDS(file = paste0("./data/ar1_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  x <- rvgaw_data$x
  phi <- rvgaw_data$phi
  sigma_e <- rvgaw_data$sigma_e
}

plot(1:n, x, type = "l")

####################################################
##         R-VGA with Whittle likelihood          ##
####################################################
S <- 200

# Priors: theta_phi ~ N(0, 1), theta_sigma ~ N(0, 0.5)
mu_0 <- 0
P_0 <- 1

print("Starting exact R-VGAL...")
rvgae.t1 <- proc.time()

prior_mean <- mu_0
prior_var <- P_0

rvgae.mu_vals <- list()
rvgae.mu_vals[[1]] <- prior_mean

rvgae.prec <- list()
rvgae.prec[[1]] <- 1/prior_var #chol2inv(chol(P_0))

print("Starting R-VGAL with Whittle likelihood...")

rvgaw.t1 <- proc.time()

# x <- series
# n <- length(x)

rvgaw.mu_vals <- list()
rvgaw.mu_vals[[1]] <- prior_mean

rvgaw.prec <- list()

param_dim <- length(prior_mean)

if (param_dim == 1 && is.null(sigma_e)) {
  stop("sigma_e is not specified")
}
if (param_dim > 1) {
  rvgaw.prec[[1]] <- chol2inv(chol(prior_var))
} else {
  rvgaw.prec[[1]] <- 1/prior_var #chol2inv(chol(P_0))
}

for (t in 1:length(x)) {
  
  cat("t =", t, "\n")
  k_fin <- floor((t-1)/2)
  series <- x[1:t]
  
  ## I. Compute the periodogram
  ## Fourier frequencies
  k_in_likelihood <- c()
  if (t <= 2) {
    k_in_likelihood <- 0
    freq <- 0
    I <- 0
    
  } else {
    k_in_likelihood <- seq(1, k_fin) 
    freq <- 2 * pi * k_in_likelihood / n
    
    ## Fourier transform of the series
    fourier_transf <- fft(series)
    periodogram <- 1/t * Mod(fourier_transf)^2
    I <- periodogram[k_in_likelihood + 1] # shift the indices because the periodogram is calculated from k = 0, whereas in Whittle likelihood we start from k = 1
    
    # cat("k =", k_in_likelihood, "\n", "I =", I, "\n")
    
  }
  
  # a_vals <- 1
  # if (use_tempering) {
  # 
  #   if (i <= n_temper) { # only temper the first n_temper observations
  #     a_vals <- temper_schedule
  #   }
  # }
  
  a <- 1

  mu_temp <- rvgaw.mu_vals[[t]]
  prec_temp <- rvgaw.prec[[t]] 
  
  for (j in 1:length(freq)) {
    # cat("j =", j,"\t")
    ## Update mu and prec using each of the freqs (can turn this to batch later)
    
    # for (v in 1:length(a_vals)) { # for each step in the tempering schedule
    #   
    #   a <- a_vals[v]
  
    if (param_dim > 1) {
      P <- solve(prec_temp) #chol2inv(chol(prec_temp))
      samples <- rmvnorm(S, mu_temp, P)
    } else {
      P <- 1/prec_temp
      samples <- rnorm(S, mu_temp, sqrt(P))
    }
    
    grads <- list()
    hessian <- list()
    
    for (s in 1:S) {
      
      theta_s <- samples[s]
      theta_phi_s <- theta_s
      phi_s <- tanh(theta_phi_s)
      theta_sigma_s <- log(sigma_e^2) # so sigma_e^2 = exp(theta_sigma_s)
      
      # if (transform == "arctanh") {
      
        # First derivative
        grad_theta_phi <- (2*cos(freq[j])*(tanh(theta_phi_s)^2 - 1) -
                             2*tanh(theta_phi_s)*(tanh(theta_phi_s)^2 - 1)) /
          (tanh(theta_phi_s)^2 - 2*cos(freq[j])*tanh(theta_phi_s) + 1) -
          I[[j]]*exp(-theta_sigma_s)*(2*cos(freq[j])*(tanh(theta_phi_s)^2 - 1) -
                                        2*tanh(theta_phi_s)*(tanh(theta_phi_s)^2 - 1))
        grad_logW <- grad_theta_phi
        
        # Second derivative
        grad2_theta_phi <- 2 * I[[j]] * exp(-theta_sigma_s) * (tanh(theta_phi_s)^2 - 1) *
          (2*cos(freq[j])*tanh(theta_phi_s) - 3*tanh(theta_phi_s)^2 + 1) -
          (4*(cos(freq[j]) - tanh(theta_phi_s))^2 * (tanh(theta_phi_s)^2 - 1)^2) /
          (tanh(theta_phi_s)^2 - 2*cos(freq[j]) * tanh(theta_phi_s) + 1)^2 -
          (2*(tanh(theta_phi_s)^2 - 1) * (2*cos(freq[j])*tanh(theta_phi_s) -
                                            3*tanh(theta_phi_s)^2 + 1)) /
          (tanh(theta_phi_s)^2 - 2*cos(freq[j])*tanh(theta_phi_s) + 1)
        
        grad2_logW <- grad2_theta_phi
      
      # } 
      grads[[s]] <- grad_logW #grad_phi_fd
      hessian[[s]] <- grad2_logW #grad_phi_2_fd #x
      
      }
      
      E_grad <- Reduce("+", grads)/ length(grads)
      E_hessian <- Reduce("+", hessian)/ length(hessian)
      
      if ((prec_temp - a * E_hessian) < 0) {
        browser()
      }
      
      prec_temp <- prec_temp - a * E_hessian
      
      mu_temp <- mu_temp + 1/prec_temp * (a * E_grad)
      
    }

  rvgaw.prec[[t+1]] <- prec_temp
  rvgaw.mu_vals[[t+1]] <- mu_temp
  
  if (t %% floor(length(x)/10) == 0) {
    cat(floor(t/length(x) * 100), "% complete \n")
  }
}

rvgaw.t2 <- proc.time()

## Posterior density
n_post_samples <- 10000

par(mfrow = c(1,1))
rvgaw.post_var <- 1/(rvgaw.prec[[length(rvgaw.mu_vals)]])
theta.post_samples <- rnorm(n_post_samples, rvgaw.mu_vals[[length(rvgaw.mu_vals)]],
                              sqrt(rvgaw.post_var))
rvgaw.post_samples_phi <- tanh(theta.post_samples)

plot(density(rvgaw.post_samples_phi), xlab = "phi", main = "Posterior density of phi")
abline(v = phi, lty = 2)

## Trajectory
par(mfrow = c(1,2))
plot(tanh(unlist(rvgaw.mu_vals)), type = "l", ylim = c(-1, 1),
     xlab = "Iteration", ylab = "mean")
abline(h = phi, lty = 2)

plot(unlist(rvgaw.prec), type = "l", #ylim = c(-1, 1),
     xlab = "Iteration", ylab = "precision")
