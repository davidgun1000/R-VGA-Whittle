run_mcmc_ar1 <- function(series, phi = NULL, sigma_e = NULL, 
                         iters = 10000, burn_in = 2000,
                         prior_mean = 0, prior_var = 1, 
                         adapt_proposal = T, use_whittle_likelihood) {
  
  
  if (use_whittle_likelihood) {
    llh_info <- "Whittle"
  } else {
    llh_info <- "exact"
  }
  
  # cat("Starting MCMC with", llh_info, "likelihood... \n")
  
  acceptProb <- c()
  accept <- rep(0, iters)
  post_samples <- c()
  
  ## Proposal variance
  D <- 1
  if (adapt_proposal) {
    scale <- 1
    target_accept <- 0.23
  }
  
  ## Samples from prior: theta = tanh^-1(phi), theta ~ N(0,1)
  # prior_mean <- 0
  # prior_var <- 1
  theta <- rnorm(10000, prior_mean, sqrt(prior_var))
  phi_samples <- tanh(theta)
  hist(phi_samples, main = "Samples from the prior of phi")
  
  ## ALternatively can use theta = logit(phi)
  # phi_samples2 <- exp(theta) / (1 + exp(theta))
  # hist(phi_samples2)
  
  ## Initial theta: 
  theta_0 <- rnorm(1, prior_mean, sqrt(prior_var))
  theta_curr <- theta_0
  phi_curr <- tanh(theta_curr)
    
  mcmc.t1 <- proc.time()
  
  ## Compute initial likelihood
  if (use_whittle_likelihood) {
    log_likelihood_curr <- calculate_whittle_likelihood(series = series, 
                                                        phi = phi_curr, 
                                                        sigma_e = sigma_e)
    
  } else {
    log_likelihood_curr <- calculate_ar1_likelihood(series = series, 
                                                    phi = phi_curr,
                                                    sigma_e = sigma_e)
    
  }
  
  ## Compute initial prior
  log_prior_curr <- dnorm(theta_0, prior_mean, sqrt(prior_var), log = T)
  
  for (i in 1:iters) {
    
    ## 1. Propose new parameter values
    theta_prop <- rnorm(1, theta_curr, sqrt(D))
    phi_prop <- tanh(theta_prop)
    
    if (i %% (iters/10) == 0) {
      cat(i/iters * 100, "% complete \n")
      cat("Proposed phi =", tanh(theta_prop), ", Current param:", tanh(theta_curr), "\n")
      cat("------------------------------------------------------------------\n")
    }
    
    ## 2. Calculate likelihood
    if (use_whittle_likelihood) {
      log_likelihood_prop <- calculate_whittle_likelihood(series = series, 
                                                          phi = phi_prop, 
                                                          sigma_e = sigma_e)
      
    } else { # use Gaussian likelihood
      log_likelihood_prop <- calculate_ar1_likelihood(series = series, 
                                                      phi = phi_prop,
                                                      sigma_e = sigma_e)
    }
    
    ## 3. Calculate prior
    log_prior_prop <- dnorm(theta_prop, prior_mean, sqrt(prior_var), log = T)
    
    ## 4. Calculate acceptance probability
    if (abs(tanh(theta_prop)) < 1) { ## if the proposed phi is within (-1, 1)
      r <- exp((log_likelihood_prop + log_prior_prop) - (log_likelihood_curr + log_prior_curr))
      a <- min(1, r)
    } else {
      a <- 0
    }
    # cat("a =", a, "\n")
    acceptProb[i] <- a
    
    ## 5. Move to next state with acceptance probability a
    u <- runif(1, 0, 1)
    if (u < a) {
      accept[i] <- 1
      phi_curr <- phi_prop
      theta_curr <- theta_prop
      log_likelihood_curr <- log_likelihood_prop
      log_prior_curr <- log_prior_prop
    } 
    
    ## Store parameter 
    post_samples[i] <- phi_curr #theta_curr
    
    ## Adapt proposal covariance matrix
    if (adapt_proposal) {
      if ((i >= 200) && (i %% 10 == 0)) {
        scale <- update_sigma(scale, a, target_accept, i, 1)
        D <- scale * var((post_samples[1:i]))
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
  
  return(mcmc_results)
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