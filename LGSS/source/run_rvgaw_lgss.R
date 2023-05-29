run_rvgaw_lgss <- function(y, phi = NULL, sigma_eta = NULL, sigma_eps = NULL, 
                          prior_mean = 0, prior_var = 1, S = 500,
                          use_tempering = T, temper_schedule = rep(1/10, 10),
                          n_temper = 100,
                          reorder_freq = F, reorder_seed = NULL,
                          decreasing = F, 
                          use_matlab_deriv = T, transform = "arctanh") {
  
  print("Starting R-VGAL with Whittle likelihood...")
  
  rvgaw.t1 <- proc.time()
  
  n <- length(y)
  
  rvgaw.mu_vals <- list()
  rvgaw.mu_vals[[1]] <- prior_mean
  
  rvgaw.prec <- list()
  rvgaw.prec[[1]] <- 1/prior_var #chol2inv(chol(P_0))
  
  ## Fourier frequencies
  k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
  k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
  freq <- 2 * pi * k_in_likelihood / n
  
  ## Fourier transform of the observations
  fourier_transf <- fft(y)
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
    
    for (v in 1:length(a_vals)) { # for each step in the tempering schedule
      
      a <- a_vals[v]
      
      P <- 1/prec_temp
      # P <- chol2inv(chol(prec_temp))
      samples <- rnorm(S, mu_temp, sqrt(P))
      
      grads <- list()
      hessian <- list()
      
      grads_old <- list()
      hessian_old <- list()
      
      # Calculate Fourier transform of the series here
      
      for (s in 1:S) {
        
        theta_s <- samples[s]
        phi_s <- tanh(theta_s)
        
        if (use_matlab_deriv) { ## MATLAB derivatives:
          
          if (transform == "arctanh") {
            # First derivative
            # grad_logW_old <- - ((2 * cos(freq[i]) - 2 * tanh(theta_s)) * (1 - tanh(theta_s)^2) ) /
            #   (1 + tanh(theta_s)^2 - 2 * tanh(theta_s) * cos(freq[i])) -
            #   I[[i]] * (1/sigma_eta^2 * (2 * tanh(theta_s) - 2 * cos(freq[i])) * (1 - tanh(theta_s)^2))
            
            grad_logW <- (sigma_eta^2*(2*cos(freq[i])*(tanh(theta_s)^2 - 1) - 
                                         2*tanh(theta_s)*(tanh(theta_s)^2 - 1))) / 
              ((sigma_eta^2/(tanh(theta_s)^2 - 2*cos(freq[i])*tanh(theta_s) + 1) + 
                  sigma_eps^2/(2*pi))*(- 2*cos(freq[i])*tanh(theta_s) + tanh(theta_s)^2 + 1)^2) - 
              (I[[i]]*sigma_eta^2*(2*cos(freq[i])*(tanh(theta_s)^2 - 1) - 
                                     2*tanh(theta_s)*(tanh(theta_s)^2 - 1))) /
              ((sigma_eta^2/(- 2*cos(freq[i])*tanh(theta_s) + tanh(theta_s)^2 + 1) + 
                  sigma_eps^2/(2*pi))^2 * (- 2*cos(freq[i])*tanh(theta_s) + tanh(theta_s)^2 + 1)^2)
            
            # Second derivative
            # grad2_logW_old <- (2*(phi_s^2 - 1)^2 + 4*phi_s^2*(phi_s^2 - 1) -
            #                  4*cos(freq[i])*phi_s*(phi_s^2 - 1))/(phi_s^2 - 2*cos(freq[i])*phi_s + 1) -
            #   (2*cos(freq[i])*(phi_s^2 - 1) - 2*phi_s*(phi_s^2 - 1))^2/
            #   (- 2*cos(freq[i])*phi_s + phi_s^2 + 1)^2 -
            #   (I[[i]]*(2*(phi_s^2 - 1)^2 + 4*phi_s^2*(phi_s^2 - 1) -
            #              4*cos(freq[i])*phi_s*(phi_s^2 - 1)))/sigma_eta^2
            
            grad2_logW <- (sigma_eta^2*(2*(tanh(theta_s)^2 - 1)^2 + 
                                          4*tanh(theta_s)^2*(tanh(theta_s)^2 - 1) - 4*cos(freq[i])*tanh(theta_s)*(tanh(theta_s)^2 - 1))) / 
              ((sigma_eta^2/(tanh(theta_s)^2 - 2*cos(freq[i])*tanh(theta_s) + 1) + 
                  sigma_eps^2/(2*pi))*(- 2*cos(freq[i])*tanh(theta_s) + tanh(theta_s)^2 + 1)^2) - 
              (2*sigma_eta^2*(2*cos(freq[i])*(tanh(theta_s)^2 - 1) - 
                                2*tanh(theta_s)*(tanh(theta_s)^2 - 1))^2) / 
              ((sigma_eta^2/(tanh(theta_s)^2 - 2*cos(freq[i])*tanh(theta_s) + 1) + 
                  sigma_eps^2/(2*pi)) * (- 2*cos(freq[i])*tanh(theta_s) + tanh(theta_s)^2 + 1)^3) + 
              (sigma_eta^4*(2*cos(freq[i]) * (tanh(theta_s)^2 - 1) - 2*tanh(theta_s)*(tanh(theta_s)^2 - 1))^2) / 
              ((sigma_eta^2/(- 2*cos(freq[i])*tanh(theta_s) + tanh(theta_s)^2 + 1) + 
                  sigma_eps^2/(2*pi))^2 * (- 2*cos(freq[i])*tanh(theta_s) + tanh(theta_s)^2 + 1)^4) + 
              (2*I[[i]]*sigma_eta^2*(2*cos(freq[i])*(tanh(theta_s)^2 - 1) - 2*tanh(theta_s)*(tanh(theta_s)^2 - 1))^2) / 
              ((sigma_eta^2/(- 2*cos(freq[i])*tanh(theta_s) + tanh(theta_s)^2 + 1) + 
                  sigma_eps^2/(2*pi))^2 * 
                 (- 2*cos(freq[i])*tanh(theta_s) + tanh(theta_s)^2 + 1)^3) - 
              (2*I[[i]]*sigma_eta^4*(2*cos(freq[i])*(tanh(theta_s)^2 - 1) - 2*tanh(theta_s)*(tanh(theta_s)^2 - 1))^2) / 
              ((sigma_eta^2/(- 2*cos(freq[i])*tanh(theta_s) + tanh(theta_s)^2 + 1) + sigma_eps^2/(2*pi))^3 * 
                 (- 2*cos(freq[i])*tanh(theta_s) + tanh(theta_s)^2 + 1)^4) - 
              (I[[i]]*sigma_eta^2*(2*(tanh(theta_s)^2 - 1)^2 + 4*tanh(theta_s)^2*(tanh(theta_s)^2 - 1) - 
                                     4*cos(freq[i])*tanh(theta_s)*(tanh(theta_s)^2 - 1))) / 
              ((sigma_eta^2/(- 2*cos(freq[i])*tanh(theta_s) + tanh(theta_s)^2 + 1) + 
                  sigma_eps^2/(2*pi))^2*(- 2*cos(freq[i])*tanh(theta_s) + tanh(theta_s)^2 + 1)^2)
            
          } else { # use logit transform
            grad_logW <- (sigma_eta^2*((2*exp(2*theta_s))/(exp(theta_s) + 1)^2 - 
                                         (2*exp(3*theta_s))/(exp(theta_s) + 1)^3 + 
                                         (2*exp(2*theta_s)*cos(freq[i]))/(exp(theta_s) + 1)^2 - 
                                         (2*exp(theta_s)*cos(freq[i]))/(exp(theta_s) + 1))) / 
                        ((sigma_eta^2/(exp(2*theta_s)/(exp(theta_s) + 1)^2 - 
                                         (2*exp(theta_s)*cos(freq[i]))/(exp(theta_s) + 1) + 1) + 
                            sigma_eps^2/(2*pi)) * 
                           (exp(2*theta_s)/(exp(theta_s) + 1)^2 - (2*exp(theta_s)*cos(freq[i]))/(exp(theta_s) + 1) + 1)^2) - 
              (I[[i]]*sigma_eta^2*((2*exp(2*theta_s))/(exp(theta_s) + 1)^2 - 
                                (2*exp(3*theta_s))/(exp(theta_s) + 1)^3 + 
                                (2*exp(2*theta_s)*cos(freq[i]))/(exp(theta_s) + 1)^2 - 
                                (2*exp(theta_s)*cos(freq[i]))/(exp(theta_s) + 1))) / 
              ((sigma_eta^2/(exp(2*theta_s)/(exp(theta_s) + 1)^2 - 
                               (2*exp(theta_s)*cos(freq[i]))/(exp(theta_s) + 1) + 1) + 
                  sigma_eps^2/(2*pi))^2 * 
                 (exp(2*theta_s)/(exp(theta_s) + 1)^2 - (2*exp(theta_s)*cos(freq[i])) / 
                    (exp(theta_s) + 1) + 1)^2)
            
            grad2_logW <- (8*I[[i]]*sigma_eta^2*exp(2*theta_s) * 
                             (cos(freq[i]) - exp(theta_s) + exp(theta_s)*cos(freq[i]))^2) / 
              ((sigma_eps^2/(2*pi) + 
                  (sigma_eta^2*(exp(theta_s) + 1)^2) / 
                  (2*exp(2*theta_s) + 2*exp(theta_s) - 2*exp(theta_s)*cos(freq[i]) - 
                     2*exp(2*theta_s)*cos(freq[i]) + 1))^2 * 
                 (2*exp(2*theta_s) + 2*exp(theta_s) - 2*exp(theta_s)*cos(freq[i]) - 
                    2*exp(2*theta_s)*cos(freq[i]) + 1)^3) - 
              (8*sigma_eta^2*exp(2*theta_s) * 
                 (cos(freq[i]) - exp(theta_s) + exp(theta_s)*cos(freq[i]))^2) / 
              ((sigma_eps^2/(2*pi) + (sigma_eta^2*(exp(theta_s) + 1)^2) / 
                  (2*exp(2*theta_s) + 2*exp(theta_s) - 2*exp(theta_s)*cos(freq[i]) - 
                     2*exp(2*theta_s)*cos(freq[i]) + 1)) * 
                 (2*exp(2*theta_s) + 2*exp(theta_s) - 2*exp(theta_s)*cos(freq[i]) - 
                    2*exp(2*theta_s)*cos(freq[i]) + 1)^3) - 
              (2*sigma_eta^2*exp(theta_s)*(exp(2*theta_s) + cos(freq[i]) - 
                                           2*exp(theta_s) - exp(2*theta_s)*cos(freq[i]))) / 
              ((sigma_eps^2/(2*pi) + (sigma_eta^2*(exp(theta_s) + 1)^2) / 
                  (2*exp(2*theta_s) + 2*exp(theta_s) - 2*exp(theta_s)*cos(freq[i]) - 
                     2*exp(2*theta_s)*cos(freq[i]) + 1)) * 
                 (2*exp(2*theta_s) + 2*exp(theta_s) - 2*exp(theta_s)*cos(freq[i]) - 
                    2*exp(2*theta_s)*cos(freq[i]) + 1)^2) + 
              (4*sigma_eta^4*exp(2*theta_s)*(exp(theta_s) + 1)^2 * 
                 (cos(freq[i]) - exp(theta_s) + exp(theta_s)*cos(freq[i]))^2) / 
              ((sigma_eps^2/(2*pi) + (sigma_eta^2*(exp(theta_s) + 1)^2) / 
                  (2*exp(2*theta_s) + 2*exp(theta_s) - 2*exp(theta_s)*cos(freq[i]) - 
                     2*exp(2*theta_s)*cos(freq[i]) + 1))^2 * 
                 (2*exp(2*theta_s) + 2*exp(theta_s) - 2*exp(theta_s)*cos(freq[i]) - 
                    2*exp(2*theta_s)*cos(freq[i]) + 1)^4) + 
              (2*I[[i]]*sigma_eta^2*exp(theta_s) * 
                 (exp(2*theta_s) + cos(freq[i]) - 2*exp(theta_s) - exp(2*theta_s)*cos(freq[i]))) / 
              ((sigma_eps^2/(2*pi) + (sigma_eta^2*(exp(theta_s) + 1)^2) / 
                  (2*exp(2*theta_s) + 2*exp(theta_s) - 2*exp(theta_s)*cos(freq[i]) - 
                     2*exp(2*theta_s)*cos(freq[i]) + 1))^2 * 
                 (2*exp(2*theta_s) + 2*exp(theta_s) - 2*exp(theta_s)*cos(freq[i]) - 
                    2*exp(2*theta_s)*cos(freq[i]) + 1)^2) - 
              (8*I[[i]]*sigma_eta^4*exp(2*theta_s)*(exp(theta_s) + 1)^2 * 
                 (cos(freq[i]) - exp(theta_s) + exp(theta_s)*cos(freq[i]))^2) / 
              ((sigma_eps^2/(2*pi) + (sigma_eta^2*(exp(theta_s) + 1)^2) / 
                  (2*exp(2*theta_s) + 2*exp(theta_s) - 2*exp(theta_s)*cos(freq[i]) - 
                     2*exp(2*theta_s)*cos(freq[i]) + 1))^3 * 
                 (2*exp(2*theta_s) + 2*exp(theta_s) - 2*exp(theta_s)*cos(freq[i]) - 
                    2*exp(2*theta_s)*cos(freq[i]) + 1)^4)
            
          }
          
        } else {
          # First derivative
          d_logspec <-  (2 * cos(freq[i]) - 2 * phi_s) * (1 - phi_s^2) /
            (1 + phi_s^2 - 2 * cos(freq[i]) * phi_s)
          d_recispec <- 1/sigma_eta^2 * (2 * phi_s - 2 * cos(freq[i])) *
            (1 - phi_s^2)
          grad_logW <- - (d_logspec + I[[i]] * d_recispec)
          
          # Second derivative
          u <- (2 * cos(freq[i]) - 2*phi_s) * (1 - phi_s^2)
          v <- 1 + phi_s^2 - 2 * phi_s * cos(freq[i])
          u_prime <- -2 * (1 - phi_s^2)^2 + (2 * cos(freq[i]) - 2 * phi_s) *
            (-2 * phi_s * (1 - phi_s^2))
          v_prime <- 2 * phi_s * (1 - phi_s^2) - 2 * cos(freq[i]) * (1 - phi_s^2)
          
          d2_logspec <- (u_prime * v - u * v_prime) / v^2
          
          d2_recispec <- 1/sigma_eta^2 * (2 * (1 - phi_s^2)^2 - 
                                          2 * (2 * phi_s - 2 * cos(freq[i])) * 
                                          (phi_s * (1 - phi_s^2))
          )
          
          grad2_logW <- - (d2_logspec + I[[i]] * d2_recispec)
        }
        
        
        grads[[s]] <- grad_logW #grad_phi_fd
        hessian[[s]] <- grad2_logW #grad_phi_2_fd #x
        
        # grads_old[[s]] <- grad_logW_old #grad_phi_fd
        # hessian_old[[s]] <- grad2_logW_old #grad_phi_2_fd #x
        
      }
      
      E_grad <- Reduce("+", grads)/ length(grads)
      E_hessian <- Reduce("+", hessian)/ length(hessian)
      
      # E_grad_old <- Reduce("+", grads_old)/ length(grads_old)
      # E_hessian_old <- Reduce("+", hessian_old)/ length(hessian_old)
      
      prec_temp <- prec_temp - a * E_hessian
      # mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * as.matrix(E_grad_logW))  
      mu_temp <- mu_temp + 1/prec_temp * (a * E_grad)
      
      stopifnot(prec_temp > 0)
      
    }  
    
    rvgaw.prec[[i+1]] <- prec_temp
    rvgaw.mu_vals[[i+1]] <- mu_temp
    
    if (i %% floor(length(freq)/10) == 0) {
      cat(floor(i/length(freq) * 100), "% complete \n")
    }
    
  }
  
  rvgaw.t2 <- proc.time()
  
  ## Posterior samples
  rvgaw.post_var <- solve(rvgaw.prec[[length(freq)]])
  
  theta.post_samples <- rnorm(10000, rvgaw.mu_vals[[length(freq)]], sqrt(rvgaw.post_var)) # these are samples of beta, log(sigma_a^2), log(sigma_e^2)
  
  if (transform == "arctanh") {
    rvgaw.post_samples <- tanh(theta.post_samples)
  } else {
    rvgaw.post_samples <- exp(theta.post_samples) / (1 + exp(theta.post_samples))
  }
  # plot(density(rvgaw.post_samples))
  
  ## Save results
  rvgaw_results <- list(mu = rvgaw.mu_vals,
                        prec = rvgaw.prec,
                        post_samples = rvgaw.post_samples,
                        transform = transform,
                        S = S,
                        use_tempering = use_tempering,
                        temper_schedule = a_vals,
                        time_elapsed = rvgaw.t2 - rvgaw.t1)
  
  return(rvgaw_results)
}