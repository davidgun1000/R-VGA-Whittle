construct_prior <- function(data) {
  
  Y <- data
  m <- nrow(Y) # dimension of VAR_m(p)
  # L_elements <- rnorm(m*(m-1)/2, 0, sqrt(0.1))
  
  # param_dim <- m^2 + m
  param_dim <- m^2 + (m*(m-1)/2 + m) # m^2 AR parameters,
                                     # m*(m-1)/2 + m parameters from the lower Cholesky factor of Sigma_eta
  
  # Prior mean
  prior_mean <- rep(0, param_dim)
  
  # Prior var for the AR parameters -- use Minnesota prior
  ar_out1 <- arima(Y[1, ], order = c(1, 0, 0))
  ar_out2 <- arima(Y[2, ], order = c(1, 0, 0))
  sigma2_estimates <- c(ar_out1$sigma2, ar_out2$sigma2)
  
  indices <- data.frame(i = rep(1:m, each = m), j = rep(1:m, m))
  
  diag_var_A <- c()
  
  l = 1
  lambda_0 = 1
  theta_0 = 0.2
  for (k in 1:nrow(indices)) {
    i <- indices[k, 1]
    j <- indices[k, 2]
    
    if (i == j) {
      diag_var_A[k] <- 1
    } else {
      diag_var_A[k] <- (lambda_0 * theta_0 / l)^2 * 
        (sigma2_estimates[i] / sigma2_estimates[j])    
    }
  }
  
  ## N(0, 0.1) prior for the lower Cholesky factor
  diag_var_L <- rep(0.1, (m*(m-1)/2 + m))
  # diag_var_Sigma <- rep(0.01, m)
  
  ## now put the prior of Phi and L together so that
  ## we have a vector of (Phi, L) parameters
  prior_var <- diag(c(diag_var_A, diag_var_L))
  # prior_var <- diag(c(diag_var_A, diag_var_Sigma))
  
  return(list(prior_mean = prior_mean, prior_var = prior_var))
}