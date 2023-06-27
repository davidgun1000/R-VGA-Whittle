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
  
  rvgaw.mu_vals <- list()
  rvgaw.mu_vals[[1]] <- prior_mean
  
  rvgaw.prec <- list()
  if (param_dim > 1) {
    rvgaw.prec[[1]] <- chol2inv(chol(prior_var))
  } else {
    rvgaw.prec[[1]] <- 1/prior_var #chol2inv(chol(P_0))
  }
  
  ## Fourier frequencies
  k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
  k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
  freq <- 2 * pi * k_in_likelihood / n
  
  ## Fourier transform of the series
  fourier_transf <- fft(x)
  periodogram <- 1/n * Mod(fourier_transf)^2
  I <- periodogram[k_in_likelihood + 1]
  
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
    
    browser()
    for (v in 1:length(a_vals)) { # for each step in the tempering schedule
      
      a <- a_vals[v]
      
      if (param_dim > 1) {
        P <- solve(prec_temp) #chol2inv(chol(prec_temp))
        samples <- rmvnorm(S, mu_temp, P)
      } else {
        stopifnot(prec_temp > 0)
        P <- 1/prec_temp
        samples <- rnorm(S, mu_temp, sqrt(P))
        if (sum(is.na(samples)) > 0) {
          browser()
        }
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
          
        } else {
          theta_s <- samples[s]
          theta_phi_s <- theta_s
          phi_s <- tanh(theta_phi_s)
          theta_sigma_s <- log(sigma_e^2) # so sigma_e^2 = exp(theta_sigma_s)
        }
        
        ## Compute gradient and Hessian
        grad_logW <- 0
        grad2_logW <- 0
        
        if (use_matlab_deriv) { ## MATLAB derivatives:
          
          if (transform == "arctanh") {
            
            # First derivative
            grad_theta_phi <- - ((2 * cos(freq[i]) - 2 * tanh(theta_phi_s)) * (1 - tanh(theta_phi_s)^2) ) /
              (1 + tanh(theta_phi_s)^2 - 2 * tanh(theta_phi_s) * cos(freq[i])) -
              I[[i]] * (1/exp(theta_sigma_s) * (2 * tanh(theta_phi_s) - 2 * cos(freq[i])) * (1 - tanh(theta_phi_s)^2))
             
            
            # grad_theta_phi_matlab <- (2*cos(freq[i])*(tanh(theta_phi_s)^2 - 1) - 
            #                             2*tanh(theta_phi_s)*(tanh(theta_phi_s)^2 - 1)) / 
            #   (tanh(theta_phi_s)^2 - 2*cos(freq[i])*tanh(theta_phi_s) + 1) - 
            #   I[[i]]*exp(-theta_sigma_s)*(2*cos(freq[i])*(tanh(theta_phi_s)^2 - 1) - 
            #                          2*tanh(theta_phi_s)*(tanh(theta_phi_s)^2 - 1))
            # 
            if (param_dim > 1) {
              grad_theta_sigma <- -1 + I[[i]] / exp(theta_sigma_s) * (1 + tanh(theta_phi_s)^2 - 
                                                  2 * tanh(theta_phi_s) * cos(freq[i]))
              
              # grad_theta_sigma_matlab <- I[[i]]*exp(-theta_sigma_s)*(tanh(theta_phi_s)^2 - 2*cos(freq[i])*tanh(theta_phi_s) + 1) - 1
              
              
              grad_logW <- c(grad_theta_phi, grad_theta_sigma)
            } else {
              grad_logW <- grad_theta_phi
            }
              
            # Second derivative
            grad2_theta_phi <- (2*(phi_s^2 - 1)^2 + 4*phi_s^2*(phi_s^2 - 1) - 
                             4*cos(freq[i])*phi_s*(phi_s^2 - 1))/(phi_s^2 - 2*cos(freq[i])*phi_s + 1) - 
              (2*cos(freq[i])*(phi_s^2 - 1) - 2*phi_s*(phi_s^2 - 1))^2/
              (- 2*cos(freq[i])*phi_s + phi_s^2 + 1)^2 - 
              (I[[i]]*(2*(phi_s^2 - 1)^2 + 4*phi_s^2*(phi_s^2 - 1) - 
                         4*cos(freq[i])*phi_s*(phi_s^2 - 1)))/exp(theta_sigma_s)
            
            # grad2_theta_phi_matlab <- 2 * I[[i]] * exp(-theta_sigma_s) * (tanh(theta_phi_s)^2 - 1) *
            #   (2*cos(freq[i])*tanh(theta_phi_s) - 3*tanh(theta_phi_s)^2 + 1) -
            #   (4*(cos(freq[i]) - tanh(theta_phi_s))^2 * (tanh(theta_phi_s)^2 - 1)^2) /
            #   (tanh(theta_phi_s)^2 - 2*cos(freq[i]) * tanh(theta_phi_s) + 1)^2 -
            #   (2*(tanh(theta_phi_s)^2 - 1) * (2*cos(freq[i])*tanh(theta_phi_s) -
            #                                     3*tanh(theta_phi_s)^2 + 1)) /
            #   (tanh(theta_phi_s)^2 - 2*cos(freq[i])*tanh(theta_phi_s) + 1)

            
            if (param_dim > 1) {
              grad2_theta_sigma <- - I[[i]] * 
                (1 + tanh(theta_phi_s)^2 - 2 * tanh(theta_phi_s) * cos(freq[i])) *
                1/exp(theta_sigma_s)
                
              # grad2_theta_sigma_matlab <- -I[[i]] * exp(-theta_sigma_s) *
              #   (tanh(theta_phi_s)^2 - 2*cos(freq[i])*tanh(theta_phi_s) + 1)
              
              grad_theta_phi_theta_sigma <- I[[i]] / exp(theta_sigma_s) * 
                (2 * tanh(theta_phi_s) - 2 * cos(freq[i])) *
                (1 - tanh(theta_phi_s)^2)
              
              # grad_theta_phi_theta_sigma_matlab <- 2 * I[[i]] * exp(-theta_sigma_s) *
              #   (cos(freq[i]) - tanh(theta_phi_s)) * (tanh(theta_phi_s)^2 - 1)
              
              grad2_logW_diag <- c(grad2_theta_phi, grad2_theta_sigma)
              grad2_logW <- diag(grad2_logW_diag)
              grad2_logW[upper.tri(grad2_logW)] <- grad_theta_phi_theta_sigma
              grad2_logW[lower.tri(grad2_logW)] <- grad_theta_phi_theta_sigma
              
            } else {
              grad2_logW <- grad2_theta_phi
            }
            
          } else {
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
          
        } else {
          # First derivative
          d_logspec <-  (2 * cos(freq[i]) - 2 * phi_s) * (1 - phi_s^2) /
            (1 + phi_s^2 - 2 * cos(freq[i]) * phi_s)
          d_recispec <- 1/sigma_e^2 * (2 * phi_s - 2 * cos(freq[i])) *
            (1 - phi_s^2)
          grad_logW <- - (d_logspec + I[[i]] * d_recispec)
          
          # Second derivative
          u <- (2 * cos(freq[i]) - 2*phi_s) * (1 - phi_s^2)
          v <- 1 + phi_s^2 - 2 * phi_s * cos(freq[i])
          u_prime <- -2 * (1 - phi_s^2)^2 + (2 * cos(freq[i]) - 2 * phi_s) *
            (-2 * phi_s * (1 - phi_s^2))
          v_prime <- 2 * phi_s * (1 - phi_s^2) - 2 * cos(freq[i]) * (1 - phi_s^2)
          
          d2_logspec <- (u_prime * v - u * v_prime) / v^2
          
          d2_recispec <- 1/sigma_e^2 * (2 * (1 - phi_s^2)^2 - 
                                          2 * (2 * phi_s - 2 * cos(freq[i])) * 
                                          (phi_s * (1 - phi_s^2))
          )
          
          grad2_logW <- - (d2_logspec + I[[i]] * d2_recispec)
        }
        
        browser()
        grads[[s]] <- grad_logW #grad_phi_fd
        hessian[[s]] <- grad2_logW #grad_phi_2_fd #x
        
      }
      
      E_grad <- Reduce("+", grads)/ length(grads)
      E_hessian <- Reduce("+", hessian)/ length(hessian)
      
      prec_temp <- prec_temp - a * E_hessian
      
      if (param_dim > 1) {
        if(sum(eigen(prec_temp)$values > 0) != param_dim) {
          browser()
        }
        mu_temp <- mu_temp + solve(prec_temp) %*% (a * as.matrix(E_grad))
        # mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * as.matrix(E_grad))
        
      } else {
        mu_temp <- mu_temp + 1/prec_temp * (a * E_grad)
      }
      
      browser()
    }  
    
    rvgaw.prec[[i+1]] <- prec_temp
    rvgaw.mu_vals[[i+1]] <- mu_temp
    
    if (i %% floor(length(freq)/10) == 0) {
      cat(floor(i/length(freq) * 100), "% complete \n")
    }
    
  }
  
  rvgaw.t2 <- proc.time()
  
  ## Posterior samples
  rvgaw.post_samples <- NULL
  if (param_dim > 1) {
    rvgaw.post_var <- solve(rvgaw.prec[[length(freq)]])
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
    rvgaw.post_var <- 1/rvgaw.prec[[length(freq)]]
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