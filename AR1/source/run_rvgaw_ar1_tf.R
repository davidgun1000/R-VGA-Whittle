run_rvgaw_ar1 <- function(series, phi = NULL, sigma_e = NULL, 
                          prior_mean = 0, prior_var = 1, 
                          n_post_samples = 10000, 
                          S = 500, deriv = "tf",
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
    
    cat("i =", i, "\n")
    
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
        P <- chol2inv(chol(prec_temp))
        
        if(!isSymmetric(P)) { browser() }
        samples <- rmvnorm(S, mu_temp, P)
      } else {
        P <- 1/prec_temp
        samples <- rnorm(S, mu_temp, sqrt(P))
      }
      
      grads <- list()
      hessian <- list()
      
      theta_phi <- samples[, 1]
      theta_sigma <- samples[, 2]
      
      if (deriv == "deriv") {
        ######################## Deriv #######################

        deriv.t1 <- proc.time()
        
        if (transform == "arctanh") {
          grad_expr <- Deriv(log_likelihood_arctanh, c("theta_phi", "theta_sigma"))
          grad2_expr <- Deriv(grad_expr, c("theta_phi", "theta_sigma"))
        } else {
          grad_expr <- Deriv(log_likelihood_logit, c("theta_phi", "theta_sigma"))
          grad2_expr <- Deriv(grad_expr, c("theta_phi", "theta_sigma"))
        }
        
        # Gradient
        grad_deriv <- mapply(grad_expr, theta_phi = theta_phi, theta_sigma = theta_sigma,
                             omega_k = freq[i], I_k = I[[i]])

        E_grad_deriv <- rowMeans(grad_deriv)

        # Hessian
        grad2_deriv <- mapply(grad2_expr, theta_phi = theta_phi, theta_sigma = theta_sigma,
                              omega_k = freq[i], I_k = I[[i]])

        E_grad2_phi_deriv <- mean(grad2_deriv[1, ])
        E_grad2_phi_sigma_deriv <- mean(grad2_deriv[2, ])
        E_grad2_sigma_phi_deriv <- mean(grad2_deriv[3, ])
        E_grad2_sigma_deriv <- mean(grad2_deriv[4, ])

        E_hessian_deriv <- matrix(c(E_grad2_phi_deriv, E_grad2_phi_sigma_deriv,
                                    E_grad2_sigma_phi_deriv, E_grad2_sigma_deriv),
                                  2, 2, byrow = T)
        deriv.t2 <- proc.time()

        E_grad <- E_grad_deriv
        E_hessian <- E_hessian_deriv


      } else if (deriv == "tf") {

        #################### Tensorflow #####################
        tf.t1 <- proc.time()

        theta_sigma_tf <- tf$Variable(theta_sigma)
        theta_phi_tf <- tf$Variable(theta_phi)
        freq_i_tf <- tf$Variable(freq[i])
        I_i_tf <- tf$Variable(I[[i]])

        if (transform == "arctanh") {
          tf_out <- compute_grad_arctanh(theta_phi_tf, theta_sigma_tf, I_i_tf, freq_i_tf)
        } else {
          tf_out <- compute_grad_logit(theta_phi_tf, theta_sigma_tf, I_i_tf, freq_i_tf)
        }
        
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

        E_hessian_tf <- matrix(c(E_grad2_phi_tf, E_grad2_phi_sigma_tf, 
                                 E_grad2_sigma_phi_tf, E_grad2_sigma_tf), 2, 2, byrow = T)

        tf.t2 <- proc.time()
        E_grad <- E_grad_tf
        E_hessian <- E_hessian_tf

        
      } else { # use analytical expressions
        t1 <- proc.time()
        for (s in 1:S) {

            theta_phi_s <- theta_phi[s]
            theta_sigma_s <- theta_sigma[s]

            ### Old stuff
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

          grads[[s]] <- grad_logW #grad_phi_fd
          hessian[[s]] <- grad2_logW #grad_phi_2_fd #x
        }

        E_grad <- Reduce("+", grads)/ length(grads)
        E_hessian <- Reduce("+", hessian)/ length(hessian)

        t2 <- proc.time()
      }
      
      # browser()
      ######################################################
      
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
    
    # if (i %% floor(length(freq)/10) == 0) {
    #   cat(floor(i/length(freq) * 100), "% complete \n")
    # }
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

log_likelihood_arctanh <- function(theta_phi, theta_sigma, omega_k, I_k) { 
  
  phi <- tanh(theta_phi)
  spec_dens <- exp(theta_sigma)/( 1 + phi^2 - 2 * phi * cos(omega_k))
  llh <- - log(spec_dens) - I_k / spec_dens
}

log_likelihood_logit <- function(theta_phi, theta_sigma, omega_k, I_k) { 
  
  phi <- exp(theta_phi) / (1 + exp(theta_phi))
  spec_dens <- exp(theta_sigma)/( 1 + phi^2 - 2 * phi * cos(omega_k))
  llh <- - log(spec_dens) - I_k / spec_dens
}

compute_grad_arctanh <- tf_function(
  testf <- function(theta_phi_s, theta_sigma_s, I_i, freq_i) {
    with (tf$GradientTape() %as% tape2, {
      with (tf$GradientTape(persistent = TRUE) %as% tape1, {

        phi_s <- tf$math$tanh(theta_phi_s)
        spec_dens <- tf$math$divide(tf$math$exp(theta_sigma_s), 1 + tf$math$square(phi_s) -
                                         tf$math$multiply(2, tf$math$multiply(phi_s, tf$math$cos(freq_i))))

        log_likelihood_tf <- - tf$math$log(spec_dens) - tf$multiply(I_i, tf$math$reciprocal(spec_dens))
        # tape1$watch(c(theta_phi_tf, theta_sigma_tf))
        # log_likelihood <- tf$math$pow(theta_phi_tf, 3) + tf$math$pow(theta_sigma_tf, 3)
        # log_likelihood <- tf$math$pow(theta_tf[, 1], 3) + tf$math$pow(theta_tf[, 2], 3)

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

compute_grad_logit <- tf_function(
  testf <- function(theta_phi_s, theta_sigma_s, I_i, freq_i) {
    with (tf$GradientTape() %as% tape2, {
      with (tf$GradientTape(persistent = TRUE) %as% tape1, {
        
        phi_s <- tf$math$divide(tf$math$exp(theta_phi_s), 1 + tf$math$exp(theta_phi_s))
        
        spec_dens <- tf$math$divide(tf$math$exp(theta_sigma_s), 1 + tf$math$square(phi_s) -
                                      tf$math$multiply(2, tf$math$multiply(phi_s, tf$math$cos(freq_i))))
        
        log_likelihood_tf <- - tf$math$log(spec_dens) - tf$multiply(I_i, tf$math$reciprocal(spec_dens))
        # tape1$watch(c(theta_phi_tf, theta_sigma_tf))
        # log_likelihood <- tf$math$pow(theta_phi_tf, 3) + tf$math$pow(theta_sigma_tf, 3)
        # log_likelihood <- tf$math$pow(theta_tf[, 1], 3) + tf$math$pow(theta_tf[, 2], 3)
        
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

# compute_grad_arctanh <- tf_function(
#   testf <- function(theta_phi_s, theta_sigma_s, I_i, freq_i) {
#     with (tf$GradientTape() %as% tape2, {
#       with (tf$GradientTape(persistent = TRUE) %as% tape1, {
#         
#         phi_s <- tf$math$divide(tf$math$exp(theta_phi_s), 1 + tf$math$exp(theta_phi_s))
#         
#         spec_dens <- tf$math$divide(tf$math$exp(theta_sigma_s), 1 + tf$math$square(phi_s) -
#                                       tf$math$multiply(2, tf$math$multiply(phi_s, tf$math$cos(freq_i))))
#         
#         log_likelihood_tf <- - tf$math$log(spec_dens) - tf$multiply(I_i, tf$math$reciprocal(spec_dens))
#         # tape1$watch(c(theta_phi_tf, theta_sigma_tf))
#         # log_likelihood <- tf$math$pow(theta_phi_tf, 3) + tf$math$pow(theta_sigma_tf, 3)
#         # log_likelihood <- tf$math$pow(theta_tf[, 1], 3) + tf$math$pow(theta_tf[, 2], 3)
#         
#       })
#       # c(grad_theta_phi, grad_theta_sigma) %<-% tape1$gradient(log_likelihood, c(theta_phi_tf, theta_sigma_tf))
#       grad_tf %<-% tape1$gradient(log_likelihood_tf, c(theta_phi_s, theta_sigma_s))
#       # grad %<-% tape1$gradient(log_likelihood, c(theta_tf[, 1], theta_tf[, 2]))
#       
#       grad_tf <- tf$reshape(grad_tf, c(2L, dim(grad_tf[[1]])))
#     })
#     grad2_tf %<-% tape2$jacobian(grad_tf, c(theta_phi_s, theta_sigma_s))
#     
#     return(list(grad = grad_tf, 
#                 hessian = grad2_tf))
#   }
# )