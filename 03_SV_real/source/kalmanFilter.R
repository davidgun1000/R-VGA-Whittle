kalmanFilter <- function(params, state_prior_mean, state_prior_var,
                         observations, iters, transform_phi = F) {
  H <- 1
  x_mean <- c()
  x_mean[1] <- state_prior_mean
  x_var <- c()
  x_var[1] <- state_prior_var
  
  y <- observations
  
  log_likelihood <- 0
  
  phi <- params$phi
  
  sigma2 <- params$sigma_eps^2 # measurement error variance
  tau2 <- params$sigma_eta^2 # process noise variance
  
  for (k in 2:iters) {
    # Forecast
    x_mean[k] <- phi * x_mean[k-1] #+ rnorm(1, 0, sqrt(tau2))
    P <- phi * x_var[k-1] * phi + tau2
    
    # Likelihood by KF equations
    # llh_mean <- H * x_mean[k]
    # llh_var <- H * P * t(H) + sigma2
    # pred_likelihood2 <- dnorm(y[k], mean = llh_mean, sd = sqrt(llh_var))
    
    # Likelihood as derived by integrating out the states x_t
    mu_x <- P/(sigma2 + P) * y[k] + sigma2/(sigma2 + P) * x_mean[k]
    sigma2_x <- 1/(1/sigma2 + 1/P)
    pred_likelihood <- log(1/sqrt(2*pi*(sigma2 + P)) *
                             exp(1/2 * (mu_x^2 / sigma2_x - y[k]^2 / sigma2 - x_mean[k]^2 / P)))
    
    # pred_likelihood <- dnorm(y[k] - x_mean[k], 0, sqrt(P + sigma2), log = T)
    log_likelihood <- log_likelihood + pred_likelihood
    
    # Analysis
    K <- P * t(H) / (H * P * t(H) + sigma2)
    pseudo_obs <- H * x_mean[k]
    x_mean[k] <- x_mean[k] + K * (y[k] - pseudo_obs)
    x_var[k] <- (1 - K * H) * P
  }
  
  return(list(kf_mean = x_mean, kf_var = x_var,
              log_likelihood = log_likelihood))
}