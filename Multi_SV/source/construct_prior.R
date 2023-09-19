construct_prior <- function(data, prior_type = "minnesota", 
                            use_cholesky = F, byrow = T) {
  
  Y <- data
  m <- nrow(Y) # dimension of VAR_m(p)
  # L_elements <- rnorm(m*(m-1)/2, 0, sqrt(0.1))
  
  if (use_cholesky) {
    param_dim <- m^2 + (m*(m-1)/2 + m) # m^2 AR parameters, 
    # m*(m-1)/2 + m parameters from the lower Cholesky factor of Sigma_eta
  } else {
    param_dim <- m^2 + m
  }
  
  # Prior mean
  # prior_mean <- rep(0, param_dim)
  prior_mean <- c(rep(0, m^2), rep(-1, m))
  
  # Prior var for the AR parameters
  if (prior_type == "minnesota") {
    sigma2_estimates <- c()
    
    for (i in 1:m) {
      ar_out <- arima(Y[i, ], order = c(1, 0, 0))
      sigma2_estimates[i] <- ar_out$sigma2
    }
    # ar_out1 <- arima(Y[1, ], order = c(1, 0, 0))
    # ar_out2 <- arima(Y[2, ], order = c(1, 0, 0))
    # sigma2_estimates <- c(ar_out1$sigma2, ar_out2$sigma2)
    
    if (byrow) {
      indices <- data.frame(i = rep(1:m, each = m), j = rep(1:m, m))
    } else {
      indices <- data.frame(i = rep(1:m, m), j = rep(1:m, each = m))
    }
    
    diag_var_A <- c()
    
    l = 1
    lambda_0 = 1 #2
    theta_0 = 0.2
    for (k in 1:nrow(indices)) {
      i <- indices[k, 1]
      j <- indices[k, 2]

      if (i == j) {
        diag_var_A[k] <- (lambda_0/l)^2
      } else {
        diag_var_A[k] <- (lambda_0 * theta_0 / l)^2 *
          (sigma2_estimates[i] / sigma2_estimates[j])
      }
    }
    
    # l = 1
    # theta_0 = 0.2 #2
    # theta_1 = 0.5
    # for (k in 1:nrow(indices)) {
    #   i <- indices[k, 1]
    #   j <- indices[k, 2]
    # 
    #   if (i == j) {
    #     diag_var_A[k] <- theta_0/l
    #   } else {
    #     diag_var_A[k] <- theta_0 * theta_1 / l *
    #       (sigma2_estimates[i] / sigma2_estimates[j])
    #   }
    # }

  } else {
    diag_var_A <- rep(1, m^2)
  }
  
  diag_var_Sigma <- 0
  if (use_cholesky) {
    ## N(0, 0.1) prior for the lower Cholesky factor
    diag_var_Sigma <- rep(0.1, m*(m-1)/2 + m)
  } else {
    # diag_var_Sigma <- rep(0.01, m)
    diag_var_Sigma <- rep(0.1, m)
  }
  
  ## now put the prior of Phi and L together so that
  ## we have a vector of (Phi, L) parameters
  prior_var <- diag(c(diag_var_A, diag_var_Sigma))
  
  return(list(prior_mean = prior_mean, prior_var = prior_var))
}