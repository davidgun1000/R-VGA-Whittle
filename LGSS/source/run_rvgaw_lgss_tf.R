run_rvgaw_lgss <- function(y, phi = NULL, sigma_eta = NULL, sigma_eps = NULL, 
                           prior_mean = 0, prior_var = 1, 
                           deriv = "tf", S = 500,
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
  rvgaw.prec[[1]] <- chol2inv(chol(prior_var))
  
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
      
      P <- chol2inv(chol(prec_temp))
      samples <- rmvnorm(S, mu_temp, P)
      theta_phi <- samples[, 1]
      theta_eta <- samples[, 2]
      theta_eps <- samples[, 3]
            
      grads <- list()
      hessian <- list()
      
      if (deriv == "deriv") {
        ######################## Deriv #######################
        
        deriv.t1 <- proc.time()
        
        grad_expr <- 0
        grad2_expr <- 0
        
        if (transform == "arctanh") {
          grad_expr <- Deriv(log_likelihood_arctanh, c("theta_phi", "theta_eta", "theta_eps"))
          grad2_expr <- Deriv(grad_expr, c("theta_phi", "theta_eta", "theta_eps"))
        } else {
          grad_expr <- Deriv(log_likelihood_logit, c("theta_phi", "theta_eta", "theta_eps"))
          grad2_expr <- Deriv(grad_expr, c("theta_phi", "theta_eta", "theta_eps"))
        }
        
        # Gradient
        grad_deriv <- mapply(grad_expr, theta_phi = theta_phi, 
                             theta_eta = theta_eta, theta_eps = theta_eps,
                             omega_k = freq[i], I_k = I[[i]])
        
        E_grad_deriv <- rowMeans(grad_deriv)
        
        # Hessian
        grad2_deriv <- mapply(grad2_expr, theta_phi = theta_phi, 
                              theta_eta = theta_eta, theta_eps = theta_eps,
                              omega_k = freq[i], I_k = I[[i]])
        
        E_grad2s <- apply(grad2_deriv, 1, mean)
        
        E_hessian_deriv <- matrix(E_grad2s, 3, 3, byrow = T)
        deriv.t2 <- proc.time()
        
        E_grad <- E_grad_deriv
        E_hessian <- E_hessian_deriv
        
      } else if (deriv == "tf") {
        tf.t1 <- proc.time()
        
        theta_phi_tf <- tf$Variable(theta_phi)
        theta_eta_tf <- tf$Variable(theta_eta)
        theta_eps_tf <- tf$Variable(theta_eps)
        
        freq_i_tf <- tf$Variable(freq[i])
        I_i_tf <- tf$Variable(I[[i]])

        if (transform == "arctanh") {
          tf_out <- compute_grad_arctanh(theta_phi_tf, theta_eta_tf, theta_eps_tf, I_i_tf, freq_i_tf)
        } else {
          tf_out <- compute_grad_logit(theta_phi_tf, theta_sigma_tf, I_i_tf, freq_i_tf)
        }
        
        grads_tf <- tf_out$grad
        hessians_tf <- tf_out$hessian
        
        ## need to then reshape these into the right grads and hessians
        ## gradients
        E_grad_tf <- rowMeans(as.matrix(grads_tf, 3L, 3L))
        
        ## batch-extract diagonals, and then extract first element of diagonal as grad2_phi(1),
        ## second element as grad2_phi(2) etc
        grad2_phi_tf <- diag(as.matrix(hessians_tf[[1]][1,,], S, S)) #grad2_phi
        grad2_phi_eta_tf <- diag(as.matrix(hessians_tf[[1]][2,,], S, S)) #grad2_phi_sigma
        grad2_phi_eps_tf <- diag(as.matrix(hessians_tf[[1]][3,,], S, S)) #grad2_phi_sigma
        
        grad2_eta_phi_tf <- diag(as.matrix(hessians_tf[[2]][1,,], S, S)) #grad2_sigma_phi
        grad2_eta_tf <- diag(as.matrix(hessians_tf[[2]][2,,], S, S)) #grad2_sigma
        grad2_eta_eps_tf <- diag(as.matrix(hessians_tf[[2]][3,,], S, S)) #grad2_sigma
        
        grad2_eps_phi_tf <- diag(as.matrix(hessians_tf[[3]][1,,], S, S)) #grad2_sigma_phi
        grad2_eps_eta_tf <- diag(as.matrix(hessians_tf[[3]][2,,], S, S)) #grad2_sigma
        grad2_eps_tf <- diag(as.matrix(hessians_tf[[3]][3,,], S, S)) #grad2_sigma
        
        # take mean of each element in Hessian, then put them together in a 2x2 matrix E_hessian
        E_grad2_phi_tf <- mean(grad2_phi_tf)
        E_grad2_eta_tf <- mean(grad2_eta_tf)
        E_grad2_eps_tf <- mean(grad2_eps_tf)
        
        E_grad2_phi_eta_tf <- mean(grad2_phi_eta_tf)
        E_grad2_phi_eps_tf <- mean(grad2_phi_eps_tf)
        E_grad2_eta_eps_tf <- mean(grad2_eta_eps_tf)
        
        E_hessian_tf <- diag(c(E_grad2_phi_tf, E_grad2_eta_tf, E_grad2_eps_tf))
        E_hessian_tf[2, 1] <- mean(grad2_phi_eta_tf)
        E_hessian_tf[3, 1] <- mean(grad2_phi_eps_tf)
        E_hessian_tf[3, 2] <- mean(grad2_eta_eps_tf)
        
        E_hessian_tf[upper.tri(E_hessian_tf)] <- t(E_hessian_tf[lower.tri(E_hessian_tf)])
          
        tf.t2 <- proc.time()
        E_grad <- E_grad_tf
        E_hessian <- E_hessian_tf
        
      } else {
        
        t1 <- proc.time()
        for (s in 1:S) {

          theta_s <- theta_phi[s]
          phi_s <- tanh(theta_phi[s])
          sigma_eta <- sqrt(exp(theta_eta[s]))
          sigma_eps <- sqrt(exp(theta_eps[s]))
          
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

          grads[[s]] <- grad_logW #grad_phi_fd
          hessian[[s]] <- grad2_logW #grad_phi_2_fd #x

        }

        t2 <- proc.time()
        
        E_grad <- Reduce("+", grads)/ length(grads)
        E_hessian <- Reduce("+", hessian)/ length(hessian)
        
      }

      # browser()
      
      prec_temp <- prec_temp - a * E_hessian
      
      if(any(eigen(prec_temp)$value < 0)) {
        browser()
      }
      
      mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * as.matrix(E_grad))
      # mu_temp <- mu_temp + 1/prec_temp * (a * E_grad)
      
      
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
  
  theta.post_samples <- rmvnorm(10000, rvgaw.mu_vals[[length(freq)]], rvgaw.post_var) # these are samples of beta, log(sigma_a^2), log(sigma_e^2)
  
  rvgaw.post_samples_phi <- 0
  if (transform == "arctanh") {
    rvgaw.post_samples_phi <- tanh(theta.post_samples[, 1])
  } else {
    rvgaw.post_samples_phi <- exp(theta.post_samples[, 1]) / (1 + exp(theta.post_samples[, 1]))
  }
  rvgaw.post_samples_eta <- sqrt(exp(theta.post_samples[, 2]))
  rvgaw.post_samples_eps <- sqrt(exp(theta.post_samples[, 3]))
  
  rvgaw.post_samples <- list(phi = rvgaw.post_samples_phi,
                             sigma_eta = rvgaw.post_samples_eta,
                             sigma_eps = rvgaw.post_samples_eps)
                             
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

log_likelihood_arctanh <- function(theta_phi, theta_eta, theta_eps, 
                                   omega_k, I_k) { 
  
  phi <- tanh(theta_phi)
  sigma_eta <- sqrt(exp(theta_eta))
  sigma_eps <- sqrt(exp(theta_eps))
  
  spec_dens_x <- sigma_eta^2/( 1 + phi^2 - 2 * phi * cos(omega_k))
  spec_dens_eps <- sigma_eps^2 #/ (2*pi)
  spec_dens_y <- spec_dens_x + spec_dens_eps
  llh <- - log(spec_dens_y) - I_k / spec_dens_y
}

compute_grad_arctanh <- tf_function(
  testf <- function(theta_phi_s, theta_eta_s, theta_eps_s, I_i, freq_i) {
    with (tf$GradientTape() %as% tape2, {
      with (tf$GradientTape(persistent = TRUE) %as% tape1, {
        
        phi_s <- tf$math$tanh(theta_phi_s)
        spec_dens_x_tf <- tf$math$divide(tf$math$exp(theta_eta_s), 1 + tf$math$square(phi_s) -
                                      tf$math$multiply(2, tf$math$multiply(phi_s, tf$math$cos(freq_i))))
        
        ## add spec_dens_eps here
        # spec_dens_eps_tf <- tf$math$divide(tf$math$exp(theta_eps_s), tf$math$multiply(2, pi))
        spec_dens_eps_tf <- tf$math$exp(theta_eps_s)
        
        ## then
        spec_dens_y_tf <- spec_dens_x_tf + spec_dens_eps_tf
        
        log_likelihood_tf <- - tf$math$log(spec_dens_y_tf) - tf$multiply(I_i, tf$math$reciprocal(spec_dens_y_tf))
        # tape1$watch(c(theta_phi_tf, theta_sigma_tf))
        # log_likelihood <- tf$math$pow(theta_phi_tf, 3) + tf$math$pow(theta_sigma_tf, 3)
        # log_likelihood <- tf$math$pow(theta_tf[, 1], 3) + tf$math$pow(theta_tf[, 2], 3)
        
      })
      # c(grad_theta_phi, grad_theta_sigma) %<-% tape1$gradient(log_likelihood, c(theta_phi_tf, theta_sigma_tf))
      grad_tf %<-% tape1$gradient(log_likelihood_tf, c(theta_phi_s, theta_eta_s, theta_eps_s))
      # grad %<-% tape1$gradient(log_likelihood, c(theta_tf[, 1], theta_tf[, 2]))
      
      grad_tf <- tf$reshape(grad_tf, c(3L, dim(grad_tf[[1]])))
    })
    grad2_tf %<-% tape2$jacobian(grad_tf, c(theta_phi_s, theta_eta_s, theta_eps_s))
    
    return(list(grad = grad_tf,
                hessian = grad2_tf))
  }
)