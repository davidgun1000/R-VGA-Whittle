run_mcmc_lgss <- function(y, #sigma_eta, sigma_eps, 
                          iters = 10000, burn_in = 1000,
                          prior_mean = rep(0, 3), prior_var = rep(1, 3), 
                          state_ini_mean = 0, state_ini_var = 1,
                          adapt_proposal = T, use_whittle_likelihood = F) {
  mcmc.t1 <- proc.time()
  
  accept <- rep(0, iters)
  acceptProb <- c()
  post_samples_theta <- matrix(NA, iters, 3)
  
  if (use_whittle_likelihood) {
    llh_info <- "Whittle"
  } else {
    llh_info <- "exact"
  }
  cat("Running MCMC with", llh_info, "likelihood... \n")
  
  ## Initial values: sample params from prior
  theta_curr <- rmvnorm(1, prior_mean, prior_var)
  
  phi_curr <- tanh(theta_curr[1])
  sigma_eta_curr <- sqrt(exp(theta_curr[2]))
  sigma_eps_curr <- sqrt(exp(theta_curr[3]))
  
  ## Proposal variance
  D <- diag(1, 3)
  if (adapt_proposal) {
    scale <- 1
    target_accept <- 0.23
  }
  
  ## Calculate initial log likelihood and log prior
  params_curr <- list(phi = phi_curr, sigma_eta = sigma_eta_curr, sigma_eps = sigma_eps_curr)
  
  if (use_whittle_likelihood) {
    log_likelihood_curr <- compute_whittle_likelihood_lgss(y = y, 
                                                           params = params_curr)
    
  } else {
    kf_curr <- compute_kf_likelihood(state_prior_mean = state_ini_mean, 
                                     state_prior_var = state_ini_var, 
                                     iters = length(y), observations = y,
                                     params = params_curr) 
    
    log_likelihood_curr <- kf_curr$log_likelihood
  }
  
  log_prior_curr <- dmvnorm(theta_curr, prior_mean, prior_var, log = T)
  
  for (i in 1:iters) {
    
    # cat("i =", i, "\n")
    
    ## 1. Propose new parameter values
    theta_prop <- rmvnorm(1, theta_curr, D)
    phi_prop <- tanh(theta_prop[1])
    sigma_eta_prop <- sqrt(exp(theta_prop[2]))
    sigma_eps_prop <- sqrt(exp(theta_prop[3]))
    params_prop <- list(phi = phi_prop, sigma_eta = sigma_eta_prop, 
                        sigma_eps = sigma_eps_prop)
    
    if (i %% (iters/10) == 0) {
      cat(i/iters * 100, "% complete \n")
      cat("------------------------------------------------------------------\n")
    }
    
    ## 2. Calculate likelihood
    if (use_whittle_likelihood) {
      log_likelihood_prop <- compute_whittle_likelihood_lgss(y = y, 
                                                             params = params_prop)    
    } else {
      kf_prop <- compute_kf_likelihood(state_prior_mean = state_ini_mean, 
                                       state_prior_var = state_ini_var, 
                                       iters = length(y), observations = y,
                                       params = params_prop) 
      
      log_likelihood_prop <- kf_prop$log_likelihood
    }
    
    
    ## 3. Calculate prior
    log_prior_prop <- dmvnorm(theta_prop, prior_mean, prior_var, log = T)
    
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
    post_samples_theta[i, ] <- unlist(params_curr)
    
    ## Adapt proposal covariance matrix
    if (adapt_proposal) {
      if ((i >= 200) && (i %% 10 == 0)) {
        scale <- update_sigma(scale, a, target_accept, i, 3)
        D <- scale * var((post_samples_theta[1:i, ]))
      }
    }
  }
  
  mcmc.t2 <- proc.time()
  
  ## Back-transform
  post_samples_phi <- post_samples_theta[, 1]
  post_samples_eta <- post_samples_theta[, 2]
  post_samples_eps <- post_samples_theta[, 3]
  mcmc.post_samples <- list(phi = post_samples_phi,
                            sigma_eta = post_samples_eta,
                            sigma_eps = post_samples_eps)
  
  return(list(post_samples = mcmc.post_samples,
              iters = iters,
              burn_in = burn_in,
              adapt_proposal = adapt_proposal,
              accept = accept,
              time_elapsed = mcmc.t2 - mcmc.t1))
}