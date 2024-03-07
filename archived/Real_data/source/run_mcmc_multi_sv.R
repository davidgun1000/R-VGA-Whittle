run_mcmc_multi_sv <- function(data, iters, burn_in, prior_mean, prior_var,
                              adapt_proposal = F, use_whittle_likelihood = T,
                              use_cholesky = F) {
  
  mcmc.t1 <- proc.time()
  
  Y <- data
  d <- as.integer(ncol(Y))
  Tfin <- nrow(Y)
  
  if (use_cholesky) {
    param_dim <- d^2 + (d*(d-1)/2 + d) # m^2 AR parameters, 
    # m*(m-1)/2 + m parameters from the lower Cholesky factor of Sigma_eta
  } else {
    param_dim <- d^2 + d
  }

  ## Set up empty vectors for acceptance, post samples etc
  accept <- rep(0, iters)
  acceptProb <- c()
  
  # prior <- construct_prior(data = Y)
  # prior_mean <- prior$prior_mean
  # prior_var <- prior$prior_var
  # 
  # param_dim <- length(prior_mean)
  
  post_samples <- list()
  post_samples_theta <- matrix(NA, iters, param_dim)
  
  ## Pre-compute the periodogram for the data
  ## Fourier frequencies
  k <- seq(-ceiling(Tfin/2)+1, floor(Tfin/2), 1)
  k_in_likelihood <- k [k >= 1 & k <= floor((Tfin-1)/2)]
  freq <- 2 * pi * k_in_likelihood / Tfin
  
  ## astsa package
  Z <- log(Y^2) - colMeans(log(Y^2))
  fft_out <- mvspec(Z, detrend = F, plot = F)
  I <- fft_out$fxx
  
  ## Initial values: sample params from prior
  theta_curr <- rmvnorm(1, prior_mean, prior_var)
  
  ### the first 4 elements will be used to construct A
  A_curr <- matrix(theta_curr[1:(d^2)], d, d, byrow = T)
  
  ### the last 3 will be used to construct L
  if (use_cholesky) {
    L <- diag(exp(theta_curr[5:6]))
    L[2,1] <- theta_curr[7]
    Sigma_eta_curr <- L %*% t(L)
  } else {
    Sigma_eta_curr <- diag(exp(theta_curr[(d^2+1):param_dim]))
  }

  ## 3. Map (A, Sigma_eta) to (Phi, Sigma_eta) using the mapping in Ansley and Kohn (1986)
  Phi_curr <- backward_map(A_curr, Sigma_eta_curr)
  params_curr <- list(Phi = Phi_curr, Sigma_eta = Sigma_eta_curr)
  
  ## 4. Calculate the initial log likelihood
  if (use_whittle_likelihood) {
    log_likelihood_curr <- compute_whittle_likelihood_multi_sv(Y = Y, fourier_freqs = freq,
                                                               periodogram = I, 
                                                               params = params_curr)$log_likelihood
  } else { 
    # nothing here yet
  }
  
  ## 5. Calculate the initial log prior
  log_prior_curr <- dmvnorm(theta_curr, prior_mean, prior_var, log = T)
  
  ## Proposal variance
  D <- diag(rep(0.01, param_dim))
  if (adapt_proposal) {
    scale <- 1
    target_accept <- 0.23
  }
  
  for (i in 1:iters) {
    
    # cat("i =", i, "\n")
    if (i %% (iters/100) == 0) {
      cat(i/iters * 100, "% complete \n")
      cat("Acceptance rate:", sum(accept)/i, "\n")
      # cat("Current params:", unlist(params_curr), "\n")
      # cat("------------------------------------------------------------------\n")
    }
    
    ## 1. Propose new parameter values
    theta_prop <- rmvnorm(1, theta_curr, D)
    
    ### the first 4 elements will be used to construct A
    A_prop <- matrix(theta_prop[1:(d^2)], d, d, byrow = T)
    
    if (use_cholesky) {
      ### the last 3 will be used to construct L
      L <- diag(exp(theta_prop[5:6]))
      L[2,1] <- theta_prop[7]
      Sigma_eta_prop <- L %*% t(L)
    } else {
      Sigma_eta_prop <- diag(exp(theta_prop[(d^2+1):param_dim]))
    }
    
    
    ## 3. Map (A, Sigma_eta) to (Phi, Sigma_eta) using the mapping in Ansley and Kohn (1986)
    Phi_prop <- backward_map(A_prop, Sigma_eta_prop)
    params_prop <- list(Phi = Phi_prop, Sigma_eta = Sigma_eta_prop)
    
    ## 2. Calculate likelihood
    if (use_whittle_likelihood) {
      log_likelihood_prop <- compute_whittle_likelihood_multi_sv(Y = Y, fourier_freqs = freq,
                                                                 periodogram = I,
                                                                 params = params_prop)$log_likelihood
    } else { 
      # nothing here yet
    }
    
    ## 3. Calculate prior
    log_prior_prop <- dmvnorm(theta_prop, prior_mean, prior_var, log = T)
    
    # cat("llh_curr = ", log_likelihood_curr, " , llh_prop = ", log_likelihood_prop, "\n")
    # cat("lprior_curr = ", log_prior_curr, " , lprior_prop = ", log_prior_prop, "\n")
    
    ## 4. Calculate acceptance probability
    r <- exp((log_likelihood_prop + log_prior_prop) - 
               (log_likelihood_curr + log_prior_curr))
    # cat("r = ", r, "\n")
    a <- min(1, r)
    acceptProb[i] <- a
    
    ## 5. Move to next state with acceptance probability a
    u <- runif(1, 0, 1)
    if (u < a) {
      accept[i] <- 1
      theta_curr <- theta_prop # transformed parameters
      params_curr <- params_prop # params on the original scale
      log_likelihood_curr <- log_likelihood_prop
      log_prior_curr <- log_prior_prop
    } 
    # cat("Acc =", accept[i], "\n")
    
    ## Store parameter 
    # if (use_whittle_likelihood) {
    post_samples_theta[i, ] <- theta_curr #c(params_curr$phi, params_curr$sigma_eta)
    post_samples[[i]] <- list(Phi = params_curr$Phi, Sigma_eta = params_curr$Sigma_eta)
    # } else {
    #   post_samples_theta[i, ] <- c(params_curr$phi, params_curr$sigma_eta, params_curr$sigma_xi)
    # }
    
    ## Adapt proposal covariance matrix
    if (adapt_proposal) {
      if ((i >= 100) && (i %% 10 == 0)) {
        scale <- update_sigma(scale, a, target_accept, i, param_dim)
        D <- scale * var((post_samples_theta[1:i, ]))
      }
    }
  }
  
  mcmc.t2 <- proc.time()
  
  return(list(post_samples = post_samples,
              iters = iters,
              burn_in = burn_in,
              adapt_proposal = adapt_proposal,
              accept = accept,
              time_elapsed = mcmc.t2 - mcmc.t1)) 
}

update_sigma <- function(sigma2, acc, p, i, d) { # function to adapt scale parameter in proposal covariance
  alpha = -qnorm(p/2);
  c = ((1-1/d)*sqrt(2*pi)*exp(alpha^2/2)/(2*alpha) + 1/(d*p*(1-p)));
  Theta = log(sqrt(sigma2));
  Theta = Theta+c*(acc-p)/max(200, i/d);
  theta = (exp(Theta));
  theta = theta^2;
  
  return(theta)
}