run_mcmc_ar1 <- function(series, #phi = NULL, sigma_e = NULL, 
                         iters = 10000, burn_in = 2000,
                         prior_mean, prior_var, 
                         adapt_proposal = T, use_whittle_likelihood) {
  
  if (use_whittle_likelihood) {
    llh_info <- "Whittle"
  } else {
    llh_info <- "exact"
  }
  
  mcmc.t1 <- proc.time()
  
  ## Adapting proposal
  D <- diag(1, 2)
  # stepsize <- 0.02
  scale <- 1
  target_accept <- 0.23
  
  ## Prior: theta = tanh^-1(phi), theta ~ N(0,1)
  # prior_mean <- c(0, -1)
  # prior_var <- diag(c(1, 0.1))
  theta <- rmvnorm(10000, prior_mean, prior_var)
  phi_samples <- tanh(theta[, 1])
  # hist(phi_samples, main = "Samples from the prior of phi")
  sigma_e_samples <- sqrt(exp(theta[, 1]))
  # hist(sigma_e_samples, main = "Samples from the prior of sigma_eta")
  
  ## ALternatively can use theta = logit(phi)
  # phi_samples2 <- exp(theta) / (1 + exp(theta))
  # hist(phi_samples2)
  
  acceptProb <- c()
  accept <- rep(0, iters)
  post_samples <- matrix(NA, nrow = iters, ncol = length(prior_mean))
  
  ## Initial theta: 
  theta_0 <- rmvnorm(1, prior_mean, prior_var)
  theta_curr <- theta_0
  phi_curr <- tanh(theta_curr[1])
  sigma_e_curr <- sqrt(exp(theta_curr[2]))
  
  ## Compute initial likelihood
  if (use_whittle_likelihood) {
    # log_likelihood_curr <- - LS.whittle.loglik(x = c(phi_curr, 0, sigma_e), 
    #                                            series = x, order = c(p = 1, q = 0),
    #                                            ar.order = 1)
    log_likelihood_curr <- calculate_whittle_likelihood(series = x, 
                                                        phi = phi_curr, 
                                                        sigma_e = sigma_e_curr)
    
  } else {
    t1 <- proc.time()
    log_likelihood_curr <- calculate_ar1_likelihood(series = x, 
                                                    phi = phi_curr,
                                                    sigma_e = sigma_e_curr)
    t2 <- proc.time()
    
    # log_likelihood_curr2 <- calculate_ar1_likelihood(series = x,
    #                                                 phi = phi_curr,
    #                                                 sigma_e = sigma_e)
    # t3 <- proc.time()
  }
  
  ## Compute initial prior
  log_prior_theta_phi_curr <- dnorm(theta_curr[1], prior_mean[1], sqrt(prior_var[1,1]), log = T)
  log_prior_theta_sigma_curr <- dnorm(theta_curr[2], prior_mean[2], sqrt(prior_var[2,2]), log = T)
  log_prior_curr <- log_prior_theta_phi_curr + log_prior_theta_sigma_curr
  
  for (i in 1:iters) {
    
    ## 1. Propose new parameter values
    theta_prop <- rmvnorm(1, theta_curr, D)
    phi_prop <- tanh(theta_prop[1])
    sigma_e_prop <- sqrt(exp(theta_prop[2]))
    
    if (i %% (iters/10) == 0) {
      cat(i/iters * 100, "% complete \n")
      cat("Proposed phi =", tanh(theta_prop[1]), ", Current param:", tanh(theta_curr[1]), "\n")
      cat("------------------------------------------------------------------\n")
    }
    
    ## 2. Calculate likelihood
    if (use_whittle_likelihood) {
      # log_likelihood_prop <- - LS.whittle.loglik(x = c(phi_prop, 0, sigma_e), 
      #                                            series = x, order = c(p = 1, q = 0),
      #                                            ar.order = 1)
      
      log_likelihood_prop <- calculate_whittle_likelihood(series = x, 
                                                          phi = phi_prop, 
                                                          sigma_e = sigma_e_prop)
      
    } else { # use Gaussian likelihood
      log_likelihood_prop <- calculate_ar1_likelihood(series = x, 
                                                      phi = phi_prop,
                                                      sigma_e = sigma_e_prop)
    }
    
    ## 3. Calculate prior
    # log_prior_prop <- dnorm(theta_prop, prior_mean, sqrt(prior_var), log = T)
    log_prior_theta_phi_prop <- dnorm(theta_prop[1], prior_mean[1], sqrt(prior_var[1,1]), log = T)
    log_prior_theta_sigma_prop <- dnorm(theta_prop[2], prior_mean[2], sqrt(prior_var[2,2]), log = T)
    log_prior_prop <- log_prior_theta_phi_prop + log_prior_theta_sigma_prop
    
    ## 4. Calculate acceptance probability
    # if (abs(tanh(theta_prop)) < 1) { ## if the proposed phi is within (-1, 1)
    r <- exp((log_likelihood_prop + log_prior_prop) - (log_likelihood_curr + log_prior_curr))
    a <- min(1, r)
    # } else {
    #   a <- 0
    # }
    # cat("a =", a, "\n")
    acceptProb[i] <- a
    
    ## 5. Move to next state with acceptance probability a
    u <- runif(1, 0, 1)
    if (u < a) {
      accept[i] <- 1
      # phi_curr <- phi_prop
      # sigma2_curr <- sigma2_prop
      # tau2_curr <- tau2_prop
      phi_curr <- phi_prop
      sigma_e_curr <- sigma_e_prop
      theta_curr <- theta_prop
      log_likelihood_curr <- log_likelihood_prop
      log_prior_curr <- log_prior_prop
    } 
    
    ## Store parameter 
    post_samples[i, ] <- c(phi_curr, sigma_e_curr) #theta_curr
    
    ## Adapt proposal covariance matrix
    if (adapt_proposal) {
      if ((i >= 200) && (i %% 10 == 0)) {
        scale <- update_sigma(scale, a, target_accept, i, 1)
        D <- scale * var((post_samples[1:i, ]))
      }
    }
  }
  
  
  mcmc.t2 <- proc.time()
  
  mcmc_results <- list(post_samples = post_samples, 
                       phi = phi, 
                       sigma_e = sigma_e,
                       likelihood = llh_info,
                       prior_mean = prior_mean,
                       prior_var = prior_var, 
                       adapt_proposal = adapt_proposal,
                       elapsed_time = mcmc.t2 - mcmc.t1)
}

## Function to adapt MCMC proposal
update_sigma <- function(sigma2, acc, p, i, d) { # function to adapt scale parameter in proposal covariance
  alpha = -qnorm(p/2);
  c = ((1-1/d)*sqrt(2*pi)*exp(alpha^2/2)/(2*alpha) + 1/(d*p*(1-p)));
  Theta = log(sqrt(sigma2));
  Theta = Theta+c*(acc-p)/max(200, i/d);
  theta = (exp(Theta));
  theta = theta^2;
  
  return(theta)
}