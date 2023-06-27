run_rvgae_ar1 <- function(series, phi = NULL, sigma_e = NULL, 
                          prior_mean = 0, prior_var = 1, 
                          deriv = "deriv", S = 500,
                          use_tempering = T, use_matlab_deriv = T) {
  print("Starting exact R-VGAL...")
  rvgae.t1 <- proc.time()
  
  x <- series
  n <- length(x)
  param_dim <- length(prior_mean)
  
  rvgae.mu_vals <- list()
  rvgae.mu_vals[[1]] <- prior_mean
  
  rvgae.prec <- list()
  if (param_dim > 1) {
    rvgae.prec[[1]] <- chol2inv(chol(prior_var))
  } else {
    rvgae.prec[[1]] <- 1/prior_var #chol2inv(chol(P_0))
  }
  
  for (t in 1:length(x)) {
    
    cat("t =", t, "\n")
    
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
      
      if (param_dim > 1) {
        P <- solve(prec_temp) #chol2inv(chol(prec_temp))
        samples <- rmvnorm(S, mu_temp, P)
        
        theta_phi <- samples[, 1]
        theta_sigma <- samples[, 2]
      } else {
        P <- 1/prec_temp
        samples <- rnorm(S, mu_temp, sqrt(P))
      }
      
      grads <- list()
      hessians <- list()
      
      # Calculate Fourier transform of the series here
      
      # for (s in 1:S) {
      #   
      #   if (param_dim > 1) {
      #     theta_phi_s <- theta_phi[s]
      #     theta_sigma_s <- theta_sigma[s]
      #     
      #     ## Calculate the spectral density of an AR(1) -- turn this into a function later
      #     if (t == 1) {
      #       
      #       # Gradient
      #       grad_theta_phi <- - exp(-theta_sigma_s) * tanh(theta_phi_s) * 
      #         (tanh(theta_phi_s)^2 - 1)*x[1]^2 - tanh(theta_phi_s)
      #       
      #       grad_theta_sigma <- - exp(-theta_sigma_s) * (tanh(theta_phi_s)^2/2 - 1/2) * 
      #         x[1]^2 - 1/2
      #       
      #       # Hessian 
      #       grad2_theta_phi <- exp(-theta_sigma_s) *(tanh(theta_phi_s)^2 - 1) * 
      #         (exp(theta_sigma_s) + 3*x[1]^2*tanh(theta_phi_s)^2 - x[1]^2)
      #       
      #       grad2_theta_sigma <- x[1]^2*exp(-theta_sigma_s)*(tanh(theta_phi_s)^2/2 - 1/2)
      #       
      #       grad_theta_phi_theta_sigma <- x[1]^2 * exp(-theta_sigma_s) * 
      #         tanh(theta_phi_s) * (tanh(theta_phi_s)^2 - 1)
      #       
      #     } else {
      #       # Gradient components
      #       grad_theta_phi <- -x[t-1]*exp(-theta_sigma_s) * (x[t] - x[t-1]*tanh(theta_phi_s)) * 
      #         (tanh(theta_phi_s)^2 - 1)
      #       
      #       grad_theta_sigma <- (exp(-theta_sigma_s)*(x[t] - x[t-1]*tanh(theta_phi_s))^2)/2 - 1/2
      #       
      #       # Hessian components
      #       grad2_theta_phi <- x[t-1]*exp(-theta_sigma_s) * (tanh(theta_phi_s)^2 - 1) * 
      #         (x[t-1] - 3*x[t-1]*tanh(theta_phi_s)^2 + 2*x[t]*tanh(theta_phi_s))
      # 
      #       grad2_theta_sigma <- -(exp(-theta_sigma_s)*(x[t] - x[t-1]*tanh(theta_phi_s))^2)/2
      #       
      #       grad_theta_phi_theta_sigma <- x[t]^2*exp(-theta_sigma_s) * tanh(theta_phi_s) * 
      #         (tanh(theta_phi_s)^2 - 1)
      # 
      #     }
      #     
      #     # Construct grad and Hessian 
      #     grad <- c(grad_theta_phi, grad_theta_sigma)
      #     
      #     grad2 <- diag(c(grad2_theta_phi, grad2_theta_sigma))
      #     grad2[upper.tri(grad2)] <- grad_theta_phi_theta_sigma
      #     grad2[lower.tri(grad2)] <- grad_theta_phi_theta_sigma
      #     
      #   } else { # if param_dim == 1 
      #     theta_s <- samples[s]
      #     phi_s <- tanh(theta_s)
      #     
      #     ## Calculate the spectral density of an AR(1) -- turn this into a function later
      #     if (t == 1) {
      #       
      #       if (use_matlab_deriv) {
      #         grad <- - tanh(theta_s) - (x[1]^2*tanh(theta_s)*(tanh(theta_s)^2 - 1))/sigma_e^2
      #         
      #         grad2 <- tanh(theta_s)^2 + (x[1]^2*(tanh(theta_s)^2 - 1)^2)/sigma_e^2 +
      #           (2*x[1]^2*tanh(theta_s)^2*(tanh(theta_s)^2 - 1))/sigma_e^2 - 1
      #         
      #       } else {
      #         grad <- - phi_s + x[t]^1 / sigma_e^2 * phi_s * (1 - phi_s^2)
      #         grad2 <- phi_s^2 - 1 + x[t]^1 / sigma_e^2 * ( (1 - phi_s^2)^2 - 2 * phi_s^2 * (1 - phi_s^2) )
      #       }
      #       
      #     } else {
      #       
      #       if (use_matlab_deriv) {
      #         
      #         grad <- -(x[t-1]*(x[t] - x[t-1]*tanh(theta_s))*(tanh(theta_s)^2 - 1))/sigma_e^2
      #         
      #         grad2 <- (2*x[t-1]*tanh(theta_s)*(x[t] - x[t-1]*tanh(theta_s))*(tanh(theta_s)^2 - 1))/sigma_e^2 - 
      #           (x[t]^2*(tanh(theta_s)^2 - 1)^2)/sigma_e^2
      #         
      #         
      #       } else {
      #         grad <- 1/sigma_e^2 * (x[t] - phi_s * x[t-1]) * x[t-1] * (1 - phi_s^2)
      #         grad2 <- - x[t-1]^2/sigma_e^2 * (1 - phi_s^2)^2 -
      #           2 * x[t-1] / sigma_e^2 * phi_s * (1 - phi_s^2) * (x[t] - phi_s * x[t-1])
      #       }
      #       
      #     }
      #   }
      #   
      #   grads[[s]] <- grad #grad_phi_fd
      #   hessians[[s]] <- grad2 #grad_phi_2_fd #x
      # }
      # 
      # E_grad <- Reduce("+", grads)/ length(grads)
      # E_hessian <- Reduce("+", hessians)/ length(hessians)
      
      ########################## Deriv #########################
      if (deriv == "deriv") {
        grad_deriv <- 0
        grad2_deriv <- 0
        
        if (t == 1) {
          grad_expr <- Deriv(log_likelihood_1, c("theta_phi", "theta_sigma"))
          grad2_expr <- Deriv(grad_expr, c("theta_phi", "theta_sigma"))
          
          grad_deriv <- mapply(grad_expr, theta_phi = theta_phi, 
                               theta_sigma = theta_sigma, x1 = x[[1]]) 
          grad2_deriv <- mapply(grad2_expr, theta_phi = theta_phi, 
                                theta_sigma = theta_sigma, x1 = x[[1]]) 
          
        } else {
          grad_expr <- Deriv(log_likelihood_t, c("theta_phi", "theta_sigma"))
          grad2_expr <- Deriv(grad_expr, c("theta_phi", "theta_sigma"))
          
          grad_deriv <- mapply(grad_expr, theta_phi = theta_phi, 
                               theta_sigma = theta_sigma, xt = x[[t]],
                               xt_minus_1 = x[[t-1]]) 
          grad2_deriv <- mapply(grad2_expr, theta_phi = theta_phi, 
                                theta_sigma = theta_sigma, xt = x[[t]],
                                xt_minus_1 = x[[t-1]])
        }  
        
        # Expected grad
        E_grad_deriv <- rowMeans(grad_deriv)
        
        # Expected Hessian
        E_grad2_phi_deriv <- mean(grad2_deriv[1, ])
        E_grad2_phi_sigma_deriv <- mean(grad2_deriv[2, ])
        E_grad2_sigma_phi_deriv <- mean(grad2_deriv[3, ])
        E_grad2_sigma_deriv <- mean(grad2_deriv[4, ])
        
        E_hessian_deriv <- matrix(c(E_grad2_phi_deriv, E_grad2_phi_sigma_deriv, 
                                    E_grad2_sigma_phi_deriv, E_grad2_sigma_deriv), 
                                  2, 2, byrow = T)
        
        E_grad <- E_grad_deriv
        E_hessian <- E_hessian_deriv
        
      } else { # use TF
        #################### Tensorflow #####################
        tf.t1 <- proc.time()
        
        theta_sigma_tf <- tf$Variable(theta_sigma)
        theta_phi_tf <- tf$Variable(theta_phi)
        
        # if (transform == "arctanh") {
        tf_out <- compute_grad_exact(theta_phi_s = theta_phi_tf, theta_sigma_s = theta_sigma_tf, 
                                     t = t, xt = x[t], xt_minus_one = x[t-1])
        # } else {
        # tf_out <- compute_grad_logit(theta_phi_tf, theta_sigma_tf, I_i_tf, freq_i_tf)
        # }
        
        grads_tf <- tf_out$grad
        hessians_tf <- tf_out$hessian
        
        ## need to then reshape these into the right grads and hessians
        ## gradients
        E_grad_tf <- rowMeans(as.matrix(grads_tf, 2L, 2L))
        
        ## batch-extract diagonals, and then extract first element of diagonal as grad2_phi(1),
        ## second element as grad2_phi(2) etc
        grad2_phi_tf <- diag(as.matrix(hessians_tf[[1]][1,,], S, S)) #grad2_phi
        grad2_phi_sigma_tf <- diag(as.matrix(hessians_tf[[1]][2,,], S, S)) #grad2_phi_sigma
        grad2_sigma_phi_tf <- diag(as.matrix(hessians_tf[[2]][1,,], S, S)) #grad2_sigma_phi
        grad2_sigma_tf <- diag(as.matrix(hessians_tf[[2]][2,,], S, S)) #grad2_sigma
        
        # take mean of each element in Hessian, then put them together in a 2x2 matrix E_hessian
        E_grad2_phi_tf <- mean(grad2_phi_tf)
        E_grad2_phi_sigma_tf <- mean(grad2_phi_sigma_tf)
        E_grad2_sigma_phi_tf <- mean(grad2_sigma_phi_tf)
        E_grad2_sigma_tf <- mean(grad2_sigma_tf)
        
        E_hessian_tf <- matrix(c(E_grad2_phi_tf, E_grad2_phi_sigma_tf, E_grad2_sigma_phi_tf, E_grad2_sigma_tf), 2, 2, byrow = T)
        
        tf.t2 <- proc.time()
        E_grad <- E_grad_tf
        E_hessian <- E_hessian_tf
        
        ########################## End TF ############################
      }
      
      prec_temp <- prec_temp - a * E_hessian
      
      if (param_dim > 1) {
        # mu_temp <- mu_temp + solve(prec_temp) %*% (a * as.matrix(E_grad))
        mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * as.matrix(E_grad))
        
        if(sum(eigen(prec_temp)$values > 0) != param_dim) {
          browser()
        }
        
      } else {
        mu_temp <- mu_temp + 1/prec_temp * (a * E_grad)
        stopifnot(prec_temp > 0)
      }
      
      
    }  
    
    rvgae.prec[[t+1]] <- prec_temp
    rvgae.mu_vals[[t+1]] <- mu_temp
    
    if (t %% floor(length(x)/10) == 0) {
      cat(floor(t/length(x) * 100), "% complete \n")
    }
    
  }
  
  rvgae.t2 <- proc.time()
  
  ## Plot posterior
  rvgae.post_samples <- NULL
  if (param_dim > 1) {
    rvgae.post_var <- solve(rvgae.prec[[length(rvgae.mu_vals)]])
    theta.post_samples <- rmvnorm(n_post_samples, rvgae.mu_vals[[length(rvgae.mu_vals)]],
                                  rvgae.post_var)
    
    rvgae.post_samples_phi <- tanh(theta.post_samples[, 1])
    rvgae.post_samples_sigma <- sqrt(exp(theta.post_samples[, 2]))
    # plot(density(rvgaw.post_samples))
    rvgae.post_samples <- list(phi = rvgae.post_samples_phi,
                               sigma = rvgae.post_samples_sigma)
    
  } else {
    rvgae.post_var <- 1/rvgae.prec[[length(rvgae.mu_vals)]]
    theta.post_samples <- rnorm(n_post_samples, rvgae.mu_vals[[length(rvgae.mu_vals)]], 
                                sqrt(rvgae.post_var)) 
    rvgae.post_samples <- tanh(theta.post_samples)
  }
  
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

log_likelihood_1 <- function(x1, theta_phi, theta_sigma) { # p(x_1 | theta)
  llh <- -1/2 * log(exp(theta_sigma) / (1-tanh(theta_phi)^2)) - 
    1/2 * (1 - tanh(theta_phi)^2) / exp(theta_sigma) * x1^2
}

log_likelihood_t <- function(xt, xt_minus_1, theta_phi, theta_sigma) {  # p(x_t | x_t-1, theta)
  llh <- -1/2 * theta_sigma - 1/(2*exp(theta_sigma)) * (xt - tanh(theta_phi)*xt_minus_1)^2
}

## TF ##
compute_grad_exact <- tf_function(
  testf <- function(theta_phi_s, theta_sigma_s, t, xt = 0, xt_minus_one = 0) {

    with (tf$GradientTape() %as% tape2, {
      with (tf$GradientTape(persistent = TRUE) %as% tape1, {

        if (t == 1) {
          log_likelihood_tf <- - tf$math$multiply(1/2, tf$math$log(tf$math$divide(tf$math$exp(theta_sigma_s),
                                                                    (1-tf$math$tanh(theta_phi_s)^2)))) -
            tf$math$multiply(1/2,
                             tf$math$multiply(tf$math$divide((1 - tf$math$square(tf$math$tanh(theta_phi_s))),
                                                             tf$math$exp(theta_sigma_s)),
                                              tf$math$square(xt)))
          
        } else {
          log_likelihood_tf <- - tf$math$multiply(1/2, theta_sigma_s) -
            tf$math$divide(tf$math$square(xt - tf$multiply(tf$math$tanh(theta_phi_s), xt_minus_one)),
                           tf$math$multiply(1/2, exp(theta_sigma_s)))
        }
      })
      # c(grad_theta_phi, grad_theta_sigma) %<-% tape1$gradient(log_likelihood, c(theta_phi_tf, theta_sigma_tf))
      grad_tf %<-% tape1$gradient(log_likelihood_tf, c(theta_phi_s, theta_sigma_s))
      # grad %<-% tape1$gradient(log_likelihood, c(theta_tf[, 1], theta_tf[, 2]))
      grad_tf <- tf$reshape(grad_tf, c(2L, dim(grad_tf[[1]])))
    })
    grad2_tf %<-% tape2$jacobian(grad_tf, c(theta_phi_s, theta_sigma_s))

    return(list(grad = grad_tf,
                hessian = grad2_tf))
  }
)
