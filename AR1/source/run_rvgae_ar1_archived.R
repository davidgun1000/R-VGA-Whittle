run_rvgae_ar1 <- function(series, phi = NULL, sigma_e = NULL, 
                          prior_mean = 0, prior_var = 1, S = 500,
                          use_tempering = T, use_matlab_deriv = T) {
  print("Starting exact R-VGAL...")
  rvgae.t1 <- proc.time()
  
  x <- series
  n <- length(x)
  
  rvgae.mu_vals <- list()
  rvgae.mu_vals[[1]] <- prior_mean
  
  rvgae.prec <- list()
  rvgae.prec[[1]] <- 1/prior_var #chol2inv(chol(P_0))
  
  for (t in 1:length(x)) {
    
    a_vals <- 1
    if (use_tempering) {
      
      if (t <= n_temper) { # only temper the first n_temper observations
        a_vals <- temper_schedule
      } 
    } 
    
    mu_temp <- rvgae.mu_vals[[t]]
    prec_temp <- rvgae.prec[[t]] 
    
    for (v in 1:length(a_vals)) {
      
      a <- a_vals[v]
      
      P <- 1/prec_temp
      # P <- chol2inv(chol(prec_temp))
      samples <- rnorm(S, mu_temp, sqrt(P))
      
      grads <- list()
      hessian <- list()
      
      # Calculate Fourier transform of the series here
      
      for (s in 1:S) {
        
        theta_s <- samples[s]
        phi_s <- tanh(theta_s)
        
        ## Calculate the spectral density of an AR(1) -- turn this into a function later
        
        if (t == 1) {
          
          if (use_matlab_deriv) {
            # 
            
            grad <- - tanh(theta_s) - (x[1]^2*tanh(theta_s)*(tanh(theta_s)^2 - 1))/sigma_e^2
            
            grad2 <- tanh(theta_s)^2 + (x[1]^2*(tanh(theta_s)^2 - 1)^2)/sigma_e^2 +
              (2*x[1]^2*tanh(theta_s)^2*(tanh(theta_s)^2 - 1))/sigma_e^2 - 1
            
          } else {
            grad <- - phi_s + x[t]^1 / sigma_e^2 * phi_s * (1 - phi_s^2)
            grad2 <- phi_s^2 - 1 + x[t]^1 / sigma_e^2 * ( (1 - phi_s^2)^2 - 2 * phi_s^2 * (1 - phi_s^2) )
          }
          
        } else {
          
          if (use_matlab_deriv) {
            # grad <- -(x[t-1]*(x[t] - x[t-1]*phi_s)*(phi_s^2 - 1))/sigma_e^2
            
            # grad2 <- (2*x[t-1]*phi_s*(x[t] - x[t-1]*phi_s)*(phi_s^2 - 1))/sigma_e^2 -
            #   (x[t-1]^2*(phi_s^2 - 1)^2)/sigma_e^2
            
            grad <- -(x[t-1]*(x[t] - x[t-1]*tanh(theta_s))*(tanh(theta_s)^2 - 1))/sigma_e^2
            
            
            grad2 <- (2*x[t-1]*tanh(theta_s)*(x[t] - x[t-1]*tanh(theta_s))*(tanh(theta_s)^2 - 1))/sigma_e^2 - 
              (x[t]^2*(tanh(theta_s)^2 - 1)^2)/sigma_e^2
            
            
          } else {
            
            grad <- 1/sigma_e^2 * (x[t] - phi_s * x[t-1]) * x[t-1] * (1 - phi_s^2)
            grad2 <- - x[t-1]^2/sigma_e^2 * (1 - phi_s^2)^2 -
              2 * x[t-1] / sigma_e^2 * phi_s * (1 - phi_s^2) * (x[t] - phi_s * x[t-1])
            
            ## Check 2nd derivative with finite difference
            # incr <- 1e-07
            # theta_add <- theta_s + incr
            # 
            # f_theta <- 1/sigma_e^2 * (x[t] - tanh(theta_s) * x[t-1]) * x[t-1] * (1 - tanh(theta_s)^2)
            # f_theta_add <- 1/sigma_e^2 * (x[t] - tanh(theta_add) * x[t-1]) * x[t-1] * (1 - tanh(theta_add)^2)
            # 
            # grad2_fd <- (f_theta_add - f_theta) / incr
            
          }
          
        }
        
        grads[[s]] <- grad #grad_phi_fd
        hessian[[s]] <- grad2 #grad_phi_2_fd #x
      }
      
      E_grad <- Reduce("+", grads)/ length(grads)
      E_hessian <- Reduce("+", hessian)/ length(hessian)
      
      prec_temp <- prec_temp - a * E_hessian
      # mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * as.matrix(E_grad_logW))  
      mu_temp <- mu_temp + 1/prec_temp * (a * E_grad)
      
      stopifnot(prec_temp > 0)
    }  
    
    rvgae.prec[[t+1]] <- prec_temp
    rvgae.mu_vals[[t+1]] <- mu_temp
    
    if (t %% floor(length(x)/10) == 0) {
      cat(floor(t/length(x) * 100), "% complete \n")
    }
    
  }
  
  rvgae.t2 <- proc.time()
  
  ## Plot posterior
  rvgae.post_var <- solve(rvgae.prec[[length(rvgae.mu_vals)]])
  rvgae.post_samples <- tanh(rnorm(10000, rvgae.mu_vals[[length(rvgae.mu_vals)]], sqrt(rvgae.post_var))) 
  
  ## Save results
  rvgae_results <- list(mu = rvgae.mu_vals,
                        prec = rvgae.prec,
                        post_samples = rvgae.post_samples,
                        S = S,
                        use_tempering = use_tempering,
                        temper_schedule = temper_schedule,
                        time_elapsed = rvgae.t2 - rvgae.t1)
  # browser()
  
}
