run_rvgaw_sv <- function(y, phi = NULL, sigma_eta = NULL, sigma_xi = NULL, 
                         prior_mean = 0, prior_var = 1, 
                         deriv = "tf", S = 500,
                         use_tempering = T, temper_schedule = rep(1/10, 10),
                         n_temper = 100,
                         reorder_freq = F, reorder_seed = NULL,
                         decreasing = F) {
  
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
  y_tilde <- log(y^2) - mean(log(y^2))
  
  fourier_transf <- fft(y_tilde)
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
    
    # cat("i =", i, "\n")
    
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
      # theta_xi <- samples[, 3]
      theta_xi <- log(pi^2/2)
      
      grads <- list()
      hessian <- list()
      
      E_grad <- 0
      E_hessian <- 0
      
      if (deriv == "deriv") {
        deriv.t1 <- proc.time()
        
        grad_expr <- 0
        grad2_expr <- 0
        
        ### Test likelihood
        # llh_deriv <- c()
        # llh_true <- c()
        # for (s in 1:S) {
        #   params_curr <- list(phi = tanh(theta_phi[s]), sigma_eta = sqrt(exp(theta_eta[s])),
        #                       sigma_xi = sqrt(exp(theta_xi[s])))
        #   llh_true[s] <- compute_whittle_likelihood_sv(y = y, params = params_curr)
        #   llh_deriv[s] <- log_likelihood_arctanh(theta_phi[s], theta_eta[s], theta_xi[s], 
        #                                          omega_k = freq[i], I_k = I[[i]])
        # }
        
        grad_expr <- Deriv(log_likelihood_arctanh, c("theta_phi", "theta_eta"))
        grad2_expr <- Deriv(grad_expr, c("theta_phi", "theta_eta"))
      
        # Gradient
        grad_deriv <- mapply(grad_expr, theta_phi = theta_phi, 
                             theta_eta = theta_eta, theta_xi = theta_xi,
                             omega_k = freq[i], I_k = I[[i]])
        
        E_grad_deriv <- rowMeans(grad_deriv)
        
        # Hessian
        grad2_deriv <- mapply(grad2_expr, theta_phi = theta_phi, 
                              theta_eta = theta_eta, theta_xi = theta_xi,
                              omega_k = freq[i], I_k = I[[i]])
        
        E_grad2s <- apply(grad2_deriv, 1, mean)
        
        E_hessian_deriv <- matrix(E_grad2s, 2, 2, byrow = T)
        deriv.t2 <- proc.time()
        
        E_grad <- E_grad_deriv
        E_hessian <- E_hessian_deriv
        
      } else { ## Tensorflow
        #############################################      
        tf.t1 <- proc.time()
        
        theta_phi_tf <- tf$Variable(theta_phi)
        theta_eta_tf <- tf$Variable(theta_eta)
        theta_xi_tf <- tf$Variable(rep(theta_xi, S))#tf$Variable(theta_xi)
        
        samples_tf <- tf$Variable(samples, dtype = "float32")
        
        freq_i_tf <- tf$Variable(freq[i])
        I_i_tf <- tf$Variable(I[[i]])
        
        tf_out_test <- compute_grad_arctanh_test(samples_tf, I_i_tf, freq_i_tf)
        grads_tf_test <- tf_out_test$grad
        hessians_tf_test <- tf_out_test$hessian
        E_grad_tf <- tf$reduce_mean(grads_tf_test, 0L)
        E_hessian_tf <- tf$reduce_mean(hessians_tf_test, 0L)
        
        # tf_out <- compute_grad_arctanh(theta_phi_tf, theta_eta_tf, theta_xi_tf, 
        #                                I_i_tf, freq_i_tf)
        # 
        # grads_tf <- tf_out$grad
        # hessians_tf <- tf_out$hessian
        # 
        # ## need to then reshape these into the right grads and hessians
        # ## gradients
        # E_grad_tf <- rowMeans(as.matrix(grads_tf, 2L, 2L))
        # 
        # ## Optimise: do all of these in TF and do one read out at the end
        # 
        # ## batch-extract diagonals, and then extract first element of diagonal as grad2_phi(1),
        # ## second element as grad2_phi(2) etc
        # grad2_phi_tf <- diag(as.matrix(hessians_tf[[1]][1,,], S, S)) #grad2_phi
        # grad2_phi_eta_tf <- diag(as.matrix(hessians_tf[[1]][2,,], S, S)) #grad2_phi_sigma
        # # grad2_phi_xi_tf <- diag(as.matrix(hessians_tf[[1]][3,,], S, S)) #grad2_phi_sigma
        # 
        # grad2_eta_phi_tf <- diag(as.matrix(hessians_tf[[2]][1,,], S, S)) #grad2_sigma_phi
        # grad2_eta_tf <- diag(as.matrix(hessians_tf[[2]][2,,], S, S)) #grad2_sigma
        # # grad2_eta_xi_tf <- diag(as.matrix(hessians_tf[[2]][3,,], S, S)) #grad2_sigma
        # 
        # # grad2_xi_phi_tf <- diag(as.matrix(hessians_tf[[3]][1,,], S, S)) #grad2_sigma_phi
        # # grad2_xi_eta_tf <- diag(as.matrix(hessians_tf[[3]][2,,], S, S)) #grad2_sigma
        # # grad2_xi_tf <- diag(as.matrix(hessians_tf[[3]][3,,], S, S)) #grad2_sigma
        # 
        # # take mean of each element in Hessian, then put them together in a 2x2 matrix E_hessian
        # E_grad2_phi_tf <- mean(grad2_phi_tf)
        # E_grad2_eta_tf <- mean(grad2_eta_tf)
        # # E_grad2_xi_tf <- mean(grad2_xi_tf)
        # 
        # E_grad2_phi_eta_tf <- mean(grad2_phi_eta_tf)
        # # E_grad2_phi_xi_tf <- mean(grad2_phi_xi_tf)
        # # E_grad2_eta_xi_tf <- mean(grad2_eta_xi_tf)
        # 
        # # E_hessian_tf <- diag(c(E_grad2_phi_tf, E_grad2_eta_tf, E_grad2_xi_tf))
        # # E_hessian_tf[2, 1] <- mean(grad2_phi_eta_tf)
        # # E_hessian_tf[3, 1] <- mean(grad2_phi_xi_tf)
        # # E_hessian_tf[3, 2] <- mean(grad2_eta_xi_tf)
        # # 
        # # E_hessian_tf[upper.tri(E_hessian_tf)] <- t(E_hessian_tf[lower.tri(E_hessian_tf)])
        # 
        # E_hessian_tf <- matrix(c(E_grad2_phi_tf, E_grad2_phi_eta_tf, 
        #                          E_grad2_phi_eta_tf, E_grad2_eta_tf), 
        #                        2, 2, byrow = T)
        # 
        tf.t2 <- proc.time()
        
        E_grad <- as.vector(E_grad_tf)
        E_hessian <- as.matrix(E_hessian_tf)
        
      }
      prec_temp <- prec_temp - a * E_hessian
      
      # if(any(eigen(prec_temp)$value < 0)) {
      #   browser()
      # }
      
      mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * E_grad)
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
  
  rvgaw.post_samples_phi <- tanh(theta.post_samples[, 1])
  rvgaw.post_samples_eta <- sqrt(exp(theta.post_samples[, 2]))
  # rvgaw.post_samples_xi <- sqrt(exp(theta.post_samples[, 3]))
  
  rvgaw.post_samples <- list(phi = rvgaw.post_samples_phi,
                             sigma_eta = rvgaw.post_samples_eta) #,
                             # sigma_xi = rvgaw.post_samples_xi)
  
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

log_likelihood_arctanh <- function(theta_phi, theta_eta, theta_xi, 
                                   omega_k, I_k) { 
  
  phi <- tanh(theta_phi)
  sigma_eta <- sqrt(exp(theta_eta))
  sigma_xi <- sqrt(exp(theta_xi))
  
  spec_dens_x <- sigma_eta^2/( 1 + phi^2 - 2 * phi * cos(omega_k))
  spec_dens_xi <- sigma_xi^2 #/ (2*pi)
  spec_dens_y <- spec_dens_x + spec_dens_xi
  llh <- - log(spec_dens_y) - I_k / spec_dens_y
}

compute_grad_arctanh <- tf_function(
  testf <- function(theta_phi, theta_eta, theta_xi, I_i, freq_i) {
    with (tf$GradientTape() %as% tape2, {
      with (tf$GradientTape(persistent = TRUE) %as% tape1, {
        
        phi <- tf$math$tanh(theta_phi)
        spec_dens_x_tf <- tf$math$divide(tf$math$exp(theta_eta), 1 + tf$math$square(phi) -
                                           tf$math$multiply(2, tf$math$multiply(phi, tf$math$cos(freq_i))))
        
        ## add spec_dens_xi here
        # spec_dens_xi_tf <- tf$math$divide(tf$math$exp(theta_xi_s), tf$math$multiply(2, pi))
        spec_dens_xi_tf <- tf$math$exp(theta_xi)
        
        ## then
        spec_dens_y_tf <- spec_dens_x_tf + spec_dens_xi_tf
        
        log_likelihood_tf <- - tf$math$log(spec_dens_y_tf) - tf$multiply(I_i, tf$math$reciprocal(spec_dens_y_tf))
        # tape1$watch(c(theta_phi_tf, theta_sigma_tf))
        # log_likelihood <- tf$math$pow(theta_phi_tf, 3) + tf$math$pow(theta_sigma_tf, 3)
        # log_likelihood <- tf$math$pow(theta_tf[, 1], 3) + tf$math$pow(theta_tf[, 2], 3)
        
      })
      # c(grad_theta_phi, grad_theta_sigma) %<-% tape1$gradient(log_likelihood, c(theta_phi_tf, theta_sigma_tf))
      grad_tf %<-% tape1$gradient(log_likelihood_tf, c(theta_phi, theta_eta))
      # grad %<-% tape1$gradient(log_likelihood, c(theta_tf[, 1], theta_tf[, 2]))
      
      grad_tf <- tf$reshape(grad_tf, c(2L, dim(grad_tf[[1]])))
    })
    grad2_tf %<-% tape2$jacobian(grad_tf, c(theta_phi, theta_eta))
    
    return(list(llh = log_likelihood_tf, 
                grad = grad_tf,
                hessian = grad2_tf))
  }
)


compute_grad_arctanh_test <- tf_function(
  testf <- function(samples_tf, I_i, freq_i) {
    with (tf$GradientTape() %as% tape2, {
      with (tf$GradientTape(persistent = TRUE) %as% tape1, {
        
        theta_phi <- samples_tf[, 1]
        theta_eta <- samples_tf[, 2]
        
        theta_xi <- tf$constant(rep(log(pi^2/2), as.integer(length(theta_phi))))
        # theta_xi <- tf$tile(tf$constant(log(pi^2/2)), c(2L, 1L))
        
        phi <- tf$math$tanh(theta_phi)
        spec_dens_x_tf <- tf$math$divide(tf$math$exp(theta_eta), 1 + tf$math$square(phi) -
                                           tf$math$multiply(2, tf$math$multiply(phi, tf$math$cos(freq_i))))
        
        ## add spec_dens_xi here
        # spec_dens_xi_tf <- tf$math$divide(tf$math$exp(theta_xi_s), tf$math$multiply(2, pi))
        spec_dens_xi_tf <- tf$math$exp(theta_xi)
        
        ## then
        spec_dens_y_tf <- spec_dens_x_tf + spec_dens_xi_tf
        
        log_likelihood_tf <- - tf$math$log(spec_dens_y_tf) - tf$multiply(I_i, tf$math$reciprocal(spec_dens_y_tf))
        # tape1$watch(c(theta_phi_tf, theta_sigma_tf))
        # log_likelihood <- tf$math$pow(theta_phi_tf, 3) + tf$math$pow(theta_sigma_tf, 3)
        # log_likelihood <- tf$math$pow(theta_tf[, 1], 3) + tf$math$pow(theta_tf[, 2], 3)
        
      })
      # c(grad_theta_phi, grad_theta_sigma) %<-% tape1$gradient(log_likelihood, c(theta_phi_tf, theta_sigma_tf))
      # grad_tf %<-% tape1$gradient(log_likelihood_tf, c(theta_phi, theta_eta))
      grad_tf %<-% tape1$gradient(log_likelihood_tf, samples_tf)
      
      # grad %<-% tape1$gradient(log_likelihood, c(theta_tf[, 1], theta_tf[, 2]))
      # grad_tf <- tf$reshape(grad_tf, c(2L, dim(grad_tf[[1]])))
    })
    grad2_tf %<-% tape2$batch_jacobian(grad_tf, samples_tf)
    # grad2_tf %<-% tape2$jacobian(grad_tf, c(theta_phi, theta_eta))
    
    return(list(llh = log_likelihood_tf, 
                grad = grad_tf,
                hessian = grad2_tf))
  }
)