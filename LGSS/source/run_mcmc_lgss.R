run_mcmc_lgss <- function(y, sigma_eta, sigma_eps, iters = 10000, burn_in = 1000,
                          prior_mean = 0, prior_var = 1, 
                          state_ini_mean = 0, state_ini_var = 1,
                          adapt_proposal = T, use_whittle_likelihood = F) {
  mcmc.t1 <- proc.time()
  
  if (use_whittle_likelihood) {
    llh_info <- "Whittle"
  } else {
    llh_info <- "exact"
  }
  cat("Running MCMC with", llh_info, "likelihood... \n")
  
  ## Initial values: sample phi from prior
  theta_phi_curr <- rnorm(1, prior_mean, sqrt(prior_var))
  phi_curr <- tanh(theta_phi_curr)
  
  ## Proposal variance
  D <- 1
  if (adapt_proposal) {
    scale <- 1
    target_accept <- 0.23
  }
  
  ## Calculate initial log likelihood and log prior
  params_curr <- list(phi = phi_curr, sigma_eta = sigma_eta, sigma_eps = sigma_eps)
  
  if (use_whittle_likelihood) {
    log_likelihood_curr <- compute_whittle_likelihood_lgss(y = y, params = params_curr)
    
  } else {
    kf_curr <- compute_kf_likelihood(state_prior_mean = state_ini_mean, 
                                     state_prior_var = state_ini_var, 
                                     iters = length(y), observations = y,
                                     params = params_curr) 
    
    log_likelihood_curr <- kf_curr$log_likelihood
  }
  
  log_prior_curr <- dnorm(theta_phi_curr, prior_mean, sqrt(prior_var), log = T)
  
  for (i in 1:iters) {
    
    ## 1. Propose new parameter values
    theta_phi_prop <- rnorm(1, theta_phi_curr, sqrt(D))
    phi_prop <- tanh(theta_phi_prop)
    params_prop <- list(phi = phi_prop, sigma_eta = sigma_eta, sigma_eps = sigma_eps)
    
    if (i %% (iters/10) == 0) {
      cat(i/iters * 100, "% complete \n")
      cat("Proposed phi =", tanh(theta_phi_prop), ", Current param:", tanh(theta_phi_curr), "\n")
      cat("------------------------------------------------------------------\n")
    }
    
    ## 2. Calculate likelihood
    if (use_whittle_likelihood) {
      log_likelihood_prop <- compute_whittle_likelihood_lgss(y = y, params = params_prop)    
    } else {
      kf_prop <- compute_kf_likelihood(state_prior_mean = state_ini_mean, 
                                       state_prior_var = state_ini_var, 
                                       iters = length(y), observations = y,
                                       params = params_prop) 
      
      log_likelihood_prop <- kf_prop$log_likelihood
    }
    
    
    ## 3. Calculate prior
    log_prior_prop <- dnorm(theta_phi_prop, prior_mean, sqrt(prior_var), log = T)
    
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
      params_curr <- params_prop
      theta_phi_curr <- theta_phi_prop
      log_likelihood_curr <- log_likelihood_prop
      log_prior_curr <- log_prior_prop
    } 
    # cat("Acc =", accept[i], "\n")
    
    ## Store parameter 
    mcmc.post_samples[i] <- params_curr$phi
    
    ## Adapt proposal covariance matrix
    if (adapt_proposal) {
      if ((i >= 200) && (i %% 10 == 0)) {
        scale <- update_sigma(scale, a, target_accept, i, 1)
        D <- scale * var((mcmc.post_samples[1:i]))
      }
    }
  }
  
  mcmc.t2 <- proc.time()
  
  return(list(post_samples = mcmc.post_samples,
              iters = iters,
              burn_in = burn_in,
              adapt_proposal = adapt_proposal,
              accept = accept,
              time_elapsed = mcmc.t2 - mcmc.t1))
}