run_mcmc_sv <- function(y, #sigma_eta, sigma_xi, 
                        iters = 10000, burn_in = 1000,
                       prior_mean = 0, prior_var = 1, 
                       state_ini_mean = 0, state_ini_var = 1, nParticles = 100, 
                       adapt_proposal = T, use_whittle_likelihood = F) {
  
  mcmc.t1 <- proc.time()
  
  accept <- rep(0, iters)
  acceptProb <- c()
  
  param_dim <- length(prior_mean)
  
  post_samples <- post_samples_theta <- matrix(NA, iters, param_dim)
  # post_samples contains posterior samples on the original scale
  # post_samples_theta contains posterior samples on the tranformed scale
  
  ## Initial values: sample params from prior
  theta_curr <- rmvnorm(1, prior_mean, prior_var)
  
  phi_curr <- tanh(theta_curr[1])
  sigma_eta_curr <- sqrt(exp(theta_curr[2]))
  sigma_xi <- sqrt(pi^2/2)
  sigma_eps <- 1
  # if (use_whittle_likelihood) {
  #   sigma_xi_curr <- sqrt(exp(theta_curr[3]))
  # } else {
  #   sigma_eps_curr <- sqrt(exp(theta_curr[3]))
  #   # Estimate sigma_xi from sigma_eps
  #   sim_eps <- rnorm(10000, 0, sigma_eps_curr)
  #   sigma_xi_curr <- sqrt(var(log(sim_eps^2)))
  # }
  
  ## Proposal variance
  D <- diag(c(1, 1))
  if (adapt_proposal) {
    scale <- 1
    target_accept <- 0.23
  }
  
  ## Calculate initial log likelihood and log prior
  
  if (use_whittle_likelihood) {
    params_curr <- list(phi = phi_curr, sigma_eta = sigma_eta_curr, sigma_xi = sigma_xi)
    log_likelihood_curr <- compute_whittle_likelihood_sv(y = y, params = params_curr)
  } else {
    # paramsKF_curr <- list(phi = phi_curr, sigma_eta = sigma_eta_curr, sigma_eps = sigma_xi_curr)
    # 
    # kf_curr <- kalmanFilter(params = paramsKF_curr, state_prior_mean = 0, state_prior_var = 1,
    #                        observations = log(y^2), iters = length(y))
    # log_likelihood_curr <- kf_curr$log_likelihood
    
    params_curr <- list(phi = phi_curr, sigma_eta = sigma_eta_curr, sigma_eps = sigma_eps,
                        sigma_xi = sigma_xi)
    pf_out <- particleFilter(y = y, N = nParticles, iniState = 0, param = params_curr)
    log_likelihood_curr <- pf_out$log_likelihood
  }
  
  log_prior_curr <- dmvnorm(theta_curr, prior_mean, prior_var, log = T)
  
  for (i in 1:iters) {
    
    # cat("i =", i, "\n")
    
    ## 1. Propose new parameter values
    theta_prop <- rmvnorm(1, theta_curr, D)
    phi_prop <- tanh(theta_prop[1])
    sigma_eta_prop <- sqrt(exp(theta_prop[2]))
    
    # sigma_xi_prop <- sqrt(pi^2/2)
    
    # if (use_whittle_likelihood) {
    #   sigma_xi_prop <- sqrt(exp(theta_prop[3]))
    # } else {
    #   sigma_eps_prop <- sqrt(exp(theta_prop[3]))
    #   
    #   # Estimate sigma_xi from sigma_eps
    #   sim_eps <- rnorm(10000, 0, sigma_eps_prop)
    #   sigma_xi_prop <- sqrt(var(log(sim_eps^2)))
    # }
    
    
    if (i %% (iters/10) == 0) {
      cat(i/iters * 100, "% complete \n")
      cat("Acceptance rate:", sum(accept)/length(accept), "\n")
      # cat("Current params:", unlist(params_curr), "\n")
      # cat("------------------------------------------------------------------\n")
    }
    
    ## 2. Calculate likelihood
    if (use_whittle_likelihood) {
      params_prop <- list(phi = phi_prop, sigma_eta = sigma_eta_prop, 
                          sigma_xi = sigma_xi)
      log_likelihood_prop <- compute_whittle_likelihood_sv(y = y, params = params_prop)    
    } else {
      # paramsKF_prop <- list(phi = phi_prop, sigma_eta = sigma_eta_prop, sigma_eps = sigma_xi_prop)
      # 
      # kf_prop <- kalmanFilter(params = paramsKF_prop, state_prior_mean = 0, state_prior_var = 1,
      #                         observations = log(y^2), iters = length(y))
      # log_likelihood_prop <- kf_prop$log_likelihood
      
      params_prop <- list(phi = phi_prop, sigma_eta = sigma_eta_prop, sigma_eps = sigma_eps,
                          sigma_xi = sigma_xi)
      pf_out <- particleFilter(y = y, N = nParticles, iniState = 0, param = params_prop)
      log_likelihood_prop <- pf_out$log_likelihood
      
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
    # if (use_whittle_likelihood) {
      post_samples_theta[i, ] <- theta_curr # !! SHOULD THIS ACTUALLY BE BASED ON THETA_CURR (THE TRANSFORMED PARAMETERS) RATHER THAN THE ORIGINAL-SCALED PARAMETERS?
      post_samples[i, ] <- c(params_curr$phi, params_curr$sigma_eta)
    # } else {
    #   post_samples_theta[i, ] <- c(params_curr$phi, params_curr$sigma_eta, params_curr$sigma_xi)
    # }
    
    ## Adapt proposal covariance matrix
    if (adapt_proposal) {
      if ((i >= 200) && (i %% 10 == 0)) {
        scale <- update_sigma(scale, a, target_accept, i, param_dim)
        D <- scale * var((post_samples_theta[1:i, ]))
      }
    }
  }
  
  mcmc.t2 <- proc.time()
  
  ## Extract samples
  post_samples_phi <- post_samples[, 1] #post_samples_theta[, 1]
  post_samples_eta <- post_samples[, 2] #post_samples_theta[, 2]
  # post_samples_xi <- post_samples_theta[, 3]
  mcmc.post_samples <- list(phi = post_samples_phi,
                            sigma_eta = post_samples_eta) #,
                            # sigma_xi = post_samples_xi)
  
  return(list(post_samples = mcmc.post_samples,
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