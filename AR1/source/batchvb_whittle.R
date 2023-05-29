## Batch VB for AR(1) model

rm(list = ls())

source("calculate_likelihood.R")

## For storing results
date <- "20230417"
result_directory <- "./results/"

## Flags
regenerate_data <- F
rerun_vb <- T
use_whittle_likelihood <- T
use_adam <- T
save_data <- F
save_vb_results <- F

## 0. Generate data
### Parameters 
phi <- 0.99
sigma_e <- 0.5
n <- 50000 # time series length

### For the result fmilename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

if (regenerate_data) {
  ## Generate AR(1) series
  x0 <- 1
  x <- c()
  x[1] <- x0
  set.seed(2023)
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

####################################
##            Batch VB            ##
####################################

if (use_whittle_likelihood) {
  llh_info <- "whittle"
} else {
  llh_info <- "exact"
}

vb_filepath <- paste0(result_directory, "vb_", llh_info, "_results_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

if (rerun_vb) {
  print("Starting VB...")
  vb.t1 <- proc.time()
  
  VB_iters <- 2000
  
  if (use_adam) {
    ## Set initial values for ADAM
    n_variational_params <- 2
    m0 <- 0
    v0 <- 0
    m <- matrix(NA, nrow = n_variational_params, ncol = VB_iters)
    v <- matrix(NA, nrow = n_variational_params, ncol = VB_iters)
    m[, 1] <- rep(m0, n_variational_params)
    v[, 1] <- rep(v0, n_variational_params)
    t1 <- 0.9 #0.75
    t2 <- 0.99 #0.9
    alpha <- 0.005 ## Play around with the learning rate, increase it
    eps <- 1e-08
  }
  
  ## 1. Initialise variational parameters
  mu_0 <- 0
  sigma_0 <- 1
  
  mu_list <- list()
  sigma_list <- list()
  
  mu_list[[1]] <- mu_0
  sigma_list[[1]] <- sigma_0
  
  lambda <- list()
  lambda[[1]] <- list(mu = mu_0, sigma = sigma_0)
  
  LB <- c()
  
  if (use_whittle_likelihood) {
    ## Fourier frequencies
    k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
    k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
    freq <- 2 * pi * k_in_likelihood / n
    
    ## Fourier transform of the series
    fourier_transf <- fft(x)
    periodogram <- 1/n * Mod(fourier_transf)^2
    I <- periodogram[k_in_likelihood + 1]
  }
  
  for (k in 2:VB_iters) {
    
    lambda <- list(mu = mu_list[[k-1]], sigma = sigma_list[[k-1]])
    
    ## 2. Sample tau ~ N(0, 1), theta = mu + sigma_0 * tau
    tau <- rnorm(1, 0, 1)
    theta <- lambda$mu + lambda$sigma * tau
    
    ## 3. Calculate the gradient of the lower bound
    ### 3.1. Calculate the gradient of the likelihood | theta
    grad_llh <- c()
    if (use_whittle_likelihood) {
      grad_llh <- - ((2 * cos(freq) - 2 * tanh(theta)) * (1 - tanh(theta)^2) ) /
        (1 + tanh(theta)^2 - 2 * tanh(theta) * cos(freq)) - 
        I * (1/sigma_e^2 * (2 * tanh(theta) - 2 * cos(freq)) * (1 - tanh(theta)^2))
      
    } else { # use exact likelihood gradient
      grad_llh[1] <- - tanh(theta) + x[1]^2 / sigma_e^2 * tanh(theta) * (1 - tanh(theta)^2)
      
      for (t in 2:length(x)) {
        grad_llh[t] <- 1/sigma_e^2 * (x[t] - tanh(theta) * x[t-1]) * x[t-1] * (1 - tanh(theta)^2)
      }
      
    }
    grad_log_likelihood <- sum(grad_llh)
    
    ### 3.1. Calculate the gradient of the prior | theta
    grad_log_prior <- - theta
    
    ## Check grad llh and grad log prior with finite difference
    # incr <- 1e-07
    # theta_add <- theta + incr
    # theta_sub <- theta - incr
    # f_theta <- calculate_ar1_likelihood(series = x, 
    #                                     phi = tanh(theta),
    #                                     sigma_e = sigma_e)
    # f_theta_add <- calculate_ar1_likelihood(series = x, 
    #                                         phi = tanh(theta_add),
    #                                         sigma_e = sigma_e)
    # grad_llh_fd <- (f_theta_add - f_theta) / incr
    # 
    # log_prior_theta <- dnorm(theta, 0, 1, log = T)
    # log_prior_theta_add <- dnorm(theta_add, 0, 1, log = T)
    # grad_log_prior_fd <- (log_prior_theta_add - log_prior_theta) / incr
    
    ### Calculate the gradients wrt mu and sigma
    grad_mu <- grad_log_likelihood + grad_log_prior
    grad_sigma <- tau * (grad_log_likelihood + grad_log_prior) + tau^2 / (lambda$sigma)
    grad_L <- c(grad_mu, grad_sigma)
    
    ## 4. Calculate the lower bound to check for convergence
    log_likelihood <- 0
    if (use_whittle_likelihood) {
      log_likelihood <- calculate_whittle_likelihood(series = x, 
                                                     phi = tanh(theta), 
                                                     sigma_e = sigma_e) 
    } else {
      log_likelihood <- calculate_ar1_likelihood(series = x, 
                                                 phi = tanh(theta),
                                                 sigma_e = sigma_e)
    }
    
    log_prior <- -1/2 * log(2*pi) - 1/2 * theta^2
    # log_prior2 <- dnorm(theta, 0, 1, log = T)
    
    LB[k-1] <- log_likelihood + log_prior + 
      1/2 * log(2*pi) + 1/2 * log(lambda$sigma^2) + 1/2 * tau^2
    
    ## 5. Update mu and sigma
    ## Set learning rate and update parameters
    if (use_adam) {
      m[, k] <- t1 * m[, k-1] + (1 - t1) * grad_L
      v[, k] <- t2 * v[, k-1] + (1 - t2) * grad_L^2
      m_hat_k <- m[, k] / (1 - t1^k)
      v_hat_k <- v[, k] / (1 - t2^k)
      
      delta <- (alpha * m_hat_k) / (sqrt(v_hat_k) + eps)
      lambda_new <- unlist(lambda) + delta
    } else {
      a_k <- 1/(10000+k)
      lambda_new <- unlist(lambda) + a_k * grad_L
    }
    
    mu_list[[k]] <- lambda_new[1] #mu_list[[t-1]] + rho * grad_mu
    sigma_list[[k]] <- lambda_new[2] #sigma_list[[t-1]] + rho * grad_sigma
    
    stopifnot(sigma_list[[k]] > 0)
    
    if (k %% (VB_iters / 10) == 0) {
      cat(k/VB_iters * 100, "% complete \n")
    }
  }
  
  vb.t2 <- proc.time()
  
  post_var <- sigma_list[[VB_iters]]
  batchvb.post_samples <- tanh(rnorm(10000, mu_list[[VB_iters]], post_var)) # these are samples of beta, log(sigma_a^2), log(sigma_e^2)
  
  ## Save results
  vb_results <- list(mu = mu_list, sigma = sigma_list, 
                     post_samples = batchvb.post_samples,
                     phi = phi, sigma_e = sigma_e,
                     lower_bound = LB, 
                     iters = VB_iters, 
                     elapsed_time = vb.t2 - vb.t1)
  
  if (save_vb_results) {
    saveRDS(vb_results, vb_filepath)
  }
} else {
  vb_results <- readRDS(vb_filepath)
}

## Plot lower bound to check for convergence
plot(vb_results$lower_bound, type = "l")

## Plot posterior
plot(density(vb_results$post_samples), col = "purple")
abline(v = phi, lty = 2)

