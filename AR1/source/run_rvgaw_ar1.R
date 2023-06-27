run_rvgaw_ar1 <- function(series, phi = NULL, sigma_e = NULL, 
                          prior_mean = 0, prior_var = 1, S = 500,
                          n_post_samples = 10000, 
                          use_tempering = T, n_temper = 100, 
                          temper_schedule = rep(1/10, 10),
                          reorder_freq = F, reorder_seed = NULL,
                          decreasing = F, 
                          use_matlab_deriv = T, transform = "arctanh") {
  
  print("Starting R-VGAL with Whittle likelihood...")
  
  rvgaw.t1 <- proc.time()
  
  x <- series
  n <- length(x)
  param_dim <- length(prior_mean)
  
  if (param_dim == 1 && is.null(sigma_e)) {
    stop("sigma_e is not specified")
  }
  
  rvgaw.mu_vals <- list()
  rvgaw.mu_vals[[1]] <- prior_mean
  
  rvgaw.prec <- list()
  if (param_dim > 1) {
    rvgaw.prec[[1]] <- chol2inv(chol(prior_var))
  } else {
    rvgaw.prec[[1]] <- 1/prior_var #chol2inv(chol(P_0))
  }
  
  ## Fourier frequencies
  k_in_likelihood <- seq(1, floor((n-1)/2)) 
  freq <- 2 * pi * k_in_likelihood / n
  
  ## Fourier transform of the series
  fourier_transf <- fft(x)
  periodogram <- 1/n * Mod(fourier_transf)^2
  I <- periodogram[k_in_likelihood + 1]
  # I <- 1/(2*pi) * periodogram[k_in_likelihood + 1]
  
  if (reorder_freq) { # randomise order of frequencies and periodogram components
    
    if (decreasing) {
      sorted_freq <- sort(freq, decreasing = T, index.return = T)
      indices <- sorted_freq$ix
      reordered_freq <- sorted_freq$x
      reordered_I <- I[indices]
    } else {
      set.seed(reorder_seed)
      indices <- sample(1:length(freq), length(freq))
      reordered_freq <- freq[indices]
      reordered_I <- I[indices]
    }
    
    # plot(reordered_freq, type = "l")
    # lines(freq, col = "red")
    freq <- reordered_freq
    I <- reordered_I 
  }
  
  for (i in 1:length(freq)) {
    
    a_vals <- 1
    if (use_tempering) {
      
      if (i <= n_temper) { # only temper the first n_temper observations
        a_vals <- temper_schedule
      } 
    } 
    
    mu_temp <- rvgaw.mu_vals[[i]]
    prec_temp <- rvgaw.prec[[i]] 
    
    for (v in 1:length(a_vals)) { # for each step in the tempering schedule
      
      a <- a_vals[v]
      
      if (param_dim > 1) {
        P <- solve(prec_temp) #chol2inv(chol(prec_temp))
        samples <- rmvnorm(S, mu_temp, P)
      } else {
        P <- 1/prec_temp
        samples <- rnorm(S, mu_temp, sqrt(P))
      }
      
      grads <- list()
      hessian <- list()
      
      # Calculate Fourier transform of the series here
      
      for (s in 1:S) {
        
        if (param_dim > 1) {
          theta_s <- samples[s, ]
          theta_phi_s <- theta_s[1]
          theta_sigma_s <- theta_s[2]
          phi_s <- tanh(theta_phi_s) 
          
          if (transform == "arctanh") {
            
            # First derivative
            grad_theta_phi <- (2*cos(freq[i])*(tanh(theta_phi_s)^2 - 1) -
                                 2*tanh(theta_phi_s)*(tanh(theta_phi_s)^2 - 1)) /
              (tanh(theta_phi_s)^2 - 2*cos(freq[i])*tanh(theta_phi_s) + 1) -
              I[[i]]*exp(-theta_sigma_s)*(2*cos(freq[i])*(tanh(theta_phi_s)^2 - 1) -
                                            2*tanh(theta_phi_s)*(tanh(theta_phi_s)^2 - 1))
            
            grad_theta_sigma <- I[[i]]*exp(-theta_sigma_s)*(tanh(theta_phi_s)^2 - 2*cos(freq[i])*tanh(theta_phi_s) + 1) - 1
            
            grad_logW <- c(grad_theta_phi, grad_theta_sigma) 
            
            # Second derivative
            grad2_theta_phi <- 2 * I[[i]] * exp(-theta_sigma_s) * (tanh(theta_phi_s)^2 - 1) *
              (2*cos(freq[i])*tanh(theta_phi_s) - 3*tanh(theta_phi_s)^2 + 1) -
              (4*(cos(freq[i]) - tanh(theta_phi_s))^2 * (tanh(theta_phi_s)^2 - 1)^2) /
              (tanh(theta_phi_s)^2 - 2*cos(freq[i]) * tanh(theta_phi_s) + 1)^2 -
              (2*(tanh(theta_phi_s)^2 - 1) * (2*cos(freq[i])*tanh(theta_phi_s) -
                                                3*tanh(theta_phi_s)^2 + 1)) /
              (tanh(theta_phi_s)^2 - 2*cos(freq[i])*tanh(theta_phi_s) + 1)
            
            grad2_theta_sigma <- -I[[i]] * exp(-theta_sigma_s) *
              (tanh(theta_phi_s)^2 - 2*cos(freq[i])*tanh(theta_phi_s) + 1)
            
            grad_theta_phi_theta_sigma <- 2 * I[[i]] * exp(-theta_sigma_s) *
              (cos(freq[i]) - tanh(theta_phi_s)) * (tanh(theta_phi_s)^2 - 1)
            
            grad2_logW_diag <- c(grad2_theta_phi, grad2_theta_sigma)
            grad2_logW <- diag(grad2_logW_diag)
            grad2_logW[upper.tri(grad2_logW)] <- grad_theta_phi_theta_sigma
            grad2_logW[lower.tri(grad2_logW)] <- grad_theta_phi_theta_sigma
          }
          
        } else {
          theta_s <- samples[s]
          theta_phi_s <- theta_s
          phi_s <- tanh(theta_phi_s)
          theta_sigma_s <- log(sigma_e^2) # so sigma_e^2 = exp(theta_sigma_s)
          
          if (transform == "arctanh") {
            
            # First derivative
            grad_theta_phi <- (2*cos(freq[i])*(tanh(theta_phi_s)^2 - 1) -
                                 2*tanh(theta_phi_s)*(tanh(theta_phi_s)^2 - 1)) /
              (tanh(theta_phi_s)^2 - 2*cos(freq[i])*tanh(theta_phi_s) + 1) -
              I[[i]]*exp(-theta_sigma_s)*(2*cos(freq[i])*(tanh(theta_phi_s)^2 - 1) -
                                            2*tanh(theta_phi_s)*(tanh(theta_phi_s)^2 - 1))
            
            grad_logW <- grad_theta_phi
            
            # Second derivative
            grad2_theta_phi <- 2 * I[[i]] * exp(-theta_sigma_s) * (tanh(theta_phi_s)^2 - 1) *
              (2*cos(freq[i])*tanh(theta_phi_s) - 3*tanh(theta_phi_s)^2 + 1) -
              (4*(cos(freq[i]) - tanh(theta_phi_s))^2 * (tanh(theta_phi_s)^2 - 1)^2) /
              (tanh(theta_phi_s)^2 - 2*cos(freq[i]) * tanh(theta_phi_s) + 1)^2 -
              (2*(tanh(theta_phi_s)^2 - 1) * (2*cos(freq[i])*tanh(theta_phi_s) -
                                                3*tanh(theta_phi_s)^2 + 1)) /
              (tanh(theta_phi_s)^2 - 2*cos(freq[i])*tanh(theta_phi_s) + 1)
            
            grad2_logW <- grad2_theta_phi
            
          } else { # use logit transform
            grad_logW <- ((2*exp(2*theta_s))/(exp(theta_s) + 1)^2 - (2*exp(3*theta_s)) / 
                            (exp(theta_s) + 1)^3 + (2*exp(2*theta_s)*cos(freq[i])) / 
                            (exp(theta_s) + 1)^2 - (2*exp(theta_s)*cos(freq[i])) / (exp(theta_s) + 1)) /
              (exp(2*theta_s)/(exp(theta_s) + 1)^2 - (2*exp(theta_s)*cos(freq[i])) / 
                 (exp(theta_s) + 1) + 1) - 
              (I[[i]]*((2*exp(2*theta_s))/(exp(theta_s) + 1)^2 - (2*exp(3*theta_s))/(exp(theta_s) + 1)^3 + 
                         (2*exp(2*theta_s)*cos(freq[i]))/(exp(theta_s) + 1)^2 - 
                         (2*exp(theta_s)*cos(freq[i]))/(exp(theta_s) + 1))) / exp(theta_sigma_s)
            
            
            grad2_logW <- ((4*exp(2*theta_s))/(exp(theta_s) + 1)^2 - (10*exp(3*theta_s)) / 
                             (exp(theta_s) + 1)^3 + (6*exp(4*theta_s))/(exp(theta_s) + 1)^4 + 
                             (6*exp(2*theta_s)*cos(freq[i]))/(exp(theta_s) + 1)^2 - 
                             (4*exp(3*theta_s)*cos(freq[i]))/(exp(theta_s) + 1)^3 - 
                             (2*exp(theta_s)*cos(freq[i]))/(exp(theta_s) + 1)) / 
              (exp(2*theta_s)/(exp(theta_s) + 1)^2 - (2*exp(theta_s)*cos(freq[i])) / (exp(theta_s) + 1) + 1) - 
              ((2*exp(2*theta_s))/(exp(theta_s) + 1)^2 - (2*exp(3*theta_s))/(exp(theta_s) + 1)^3 + 
                 (2*exp(2*theta_s)*cos(freq[i]))/(exp(theta_s) + 1)^2 - 
                 (2*exp(theta_s)*cos(freq[i])) / (exp(theta_s) + 1))^2 / 
              (exp(2*theta_s)/(exp(theta_s) + 1)^2 - (2*exp(theta_s)*cos(freq[i])) / (exp(theta_s) + 1) + 1)^2 - 
              (I[[i]]*((4*exp(2*theta_s))/(exp(theta_s) + 1)^2 - (10*exp(3*theta_s))/(exp(theta_s) + 1)^3 + 
                         (6*exp(4*theta_s))/(exp(theta_s) + 1)^4 + 
                         (6*exp(2*theta_s)*cos(freq[i]))/(exp(theta_s) + 1)^2 - 
                         (4*exp(3*theta_s)*cos(freq[i]))/(exp(theta_s) + 1)^3 - 
                         (2*exp(theta_s)*cos(freq[i]))/(exp(theta_s) + 1)))/exp(theta_sigma_s)
            
          }
        }
        
        grads[[s]] <- grad_logW #grad_phi_fd
        hessian[[s]] <- grad2_logW #grad_phi_2_fd #x
        
      }
      
      E_grad <- Reduce("+", grads)/ length(grads)
      E_hessian <- Reduce("+", hessian)/ length(hessian)
      
      prec_temp <- prec_temp - a * E_hessian
      if(sum(eigen(prec_temp)$values > 0) != param_dim) {
        browser()
      }
      
      if (param_dim > 1) {
        # mu_temp <- mu_temp + solve(prec_temp) %*% (a * as.matrix(E_grad))
        mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * as.matrix(E_grad))
        
      } else {
        mu_temp <- mu_temp + 1/prec_temp * (a * E_grad)
      }
      
      
    }  
    
    rvgaw.prec[[i+1]] <- prec_temp
    rvgaw.mu_vals[[i+1]] <- mu_temp
    
    if (i %% floor(length(freq)/10) == 0) {
      cat(floor(i/length(freq) * 100), "% complete \n")
    }
    
  }
  
  rvgaw.t2 <- proc.time()
  
  ## Posterior samples
  rvgaw.post_var <- chol2inv(chol(rvgaw.prec[[length(freq)]]))
  
  rvgaw.post_samples <- NULL
  if (param_dim > 1) {
    theta.post_samples <- rmvnorm(n_post_samples, rvgaw.mu_vals[[length(freq)]],
                                  rvgaw.post_var)
    
    if (transform == "arctanh") {
      rvgaw.post_samples_phi <- tanh(theta.post_samples[, 1])
    } else {
      rvgaw.post_samples_phi <- exp(theta.post_samples[, 1]) / (1 + exp(theta.post_samples[, 1]))
    }
    
    rvgaw.post_samples_sigma <- sqrt(exp(theta.post_samples[, 2]))
    # plot(density(rvgaw.post_samples))
    rvgaw.post_samples <- list(phi = rvgaw.post_samples_phi,
                               sigma = rvgaw.post_samples_sigma)
    
  } else {
    theta.post_samples <- rnorm(n_post_samples, rvgaw.mu_vals[[length(freq)]], 
                                sqrt(rvgaw.post_var)) # these are samples of beta, log(sigma_a^2), log(sigma_e^2)
    if (transform == "arctanh") {
      rvgaw.post_samples <- tanh(theta.post_samples)
    } else {
      rvgaw.post_samples <- exp(theta.post_samples) / (1 + exp(theta.post_samples))
    }
  }
  
  ## Save results
  rvgaw_results <- list(mu = rvgaw.mu_vals,
                        prec = rvgaw.prec,
                        transform = transform,
                        post_samples = rvgaw.post_samples,
                        S = S,
                        use_tempering = use_tempering,
                        temper_schedule = a_vals,
                        time_elapsed = rvgaw.t2 - rvgaw.t1)
  
  return(rvgaw_results)
}