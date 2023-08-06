run_rvgaw_lgss <- function(y, phi = NULL, sigma_eta = NULL, sigma_eps = NULL, 
                           prior_mean = 0, prior_var = 1, 
                           deriv = "tf", S = 500,
                           use_tempering = T, temper_schedule = rep(1/10, 10),
                           n_temper = 100,
                           reorder_freq = F, reorder_seed = NULL,
                           decreasing = F, 
                           use_matlab_deriv = T, transform = "arctanh") {
  
  print("Starting R-VGA with Whittle likelihood...")
  
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
  
  LB <- c()
  
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
    
    log_likelihood <- c()
    
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
        
        # if (transform == "arctanh") {
          grad_expr <- Deriv(log_likelihood_arctanh, c("theta_phi", "theta_eta", "theta_eps"))
          grad2_expr <- Deriv(grad_expr, c("theta_phi", "theta_eta", "theta_eps"))
        # } else {
        #   grad_expr <- Deriv(log_likelihood_logit, c("theta_phi", "theta_eta", "theta_eps"))
        #   grad2_expr <- Deriv(grad_expr, c("theta_phi", "theta_eta", "theta_eps"))
        # }
        
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
        
      } else { # use Tensorflow to compute the grad and hessian
        tf.t1 <- proc.time()
        
        samples_tf <- tf$Variable(samples, dtype = "float32")
        # theta_phi_tf <- tf$Variable(theta_phi)
        # theta_eta_tf <- tf$Variable(theta_eta)
        # theta_eps_tf <- tf$Variable(theta_eps)
        
        freq_i_tf <- tf$constant(freq[i])
        I_i_tf <- tf$constant(I[[i]])

        tf_out_test <- compute_grad_arctanh_test(samples_tf, I_i_tf, freq_i_tf)
        E_grad_tf <- tf_out_test$E_grad
        E_hessian_tf <- tf_out_test$E_hessian
        
        # grads_tf_test <- tf_out_test$grad
        # hessians_tf_test <- tf_out_test$hessian
        # E_grad_tf <- tf$reduce_mean(grads_tf_test, 0L)
        # E_hessian_tf <- tf$reduce_mean(hessians_tf_test, 0L)
        
        # if (transform == "arctanh") {
          # tf_out <- compute_grad_arctanh(theta_phi_tf, theta_eta_tf, theta_eps_tf, I_i_tf, freq_i_tf)
        # } else {
        #   tf_out <- compute_grad_logit(theta_phi_tf, theta_sigma_tf, I_i_tf, freq_i_tf)
        # }
        # 
        # grads_tf <- tf_out$grad
        # hessians_tf <- tf_out$hessian
        # 
        # ## need to then reshape these into the right grads and hessians
        # E_grad_tf <- rowMeans(as.matrix(grads_tf, 3L, 3L))
        # 
        # ## batch-extract diagonals, and then extract first element of diagonal as grad2_phi(1),
        # ## second element as grad2_phi(2) etc
        # grad2_phi_tf <- tf$linalg$diag_part(hessians_tf[[1]])
        # grad2_eta_tf <- tf$linalg$diag_part(hessians_tf[[2]])
        # grad2_eps_tf <- tf$linalg$diag_part(hessians_tf[[3]])
        # 
        # E_grad2_phi_tf <- tf$math$reduce_mean(grad2_phi_tf, 1L) #rowMeans(as.matrix(grad2_phi))
        # E_grad2_eta_tf <- tf$math$reduce_mean(grad2_eta_tf, 1L) #rowMeans(as.matrix(grad2_eta))
        # E_grad2_eps_tf <- tf$math$reduce_mean(grad2_eps_tf, 1L) #rowMeans(as.matrix(grad2_eps))
        # 
        # E_hessian_tf <- tf$stack(list(E_grad2_phi_tf, E_grad2_eta_tf, E_grad2_eps_tf), 1L)
        # E_hessian_tf <- as.matrix(E_hessian_tf)
        
        tf.t2 <- proc.time()
        
        E_grad <- as.array(E_grad_tf)
        E_hessian <- as.array(E_hessian_tf)
        
      }
      
      ## Update variational mean and precision
      
      prec_temp <- prec_temp - a * E_hessian
      
      if(any(eigen(prec_temp)$value < 0)) {
        browser()
      }
      
      mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * E_grad)
      
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
                        lower_bound = LB,
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
    log_likelihood_tf <- 0
    with (tf$GradientTape() %as% tape2, {
      with (tf$GradientTape(persistent = TRUE) %as% tape1, {
        
        phi_s <- tf$math$tanh(theta_phi_s)
        spec_dens_x_tf <- tf$math$divide(tf$math$exp(theta_eta_s), 1 + tf$math$square(phi_s) -
                                      tf$math$multiply(2, tf$math$multiply(phi_s, tf$math$cos(freq_i))))
        
        ## add spec_dens_eps here
        spec_dens_eps_tf <- tf$math$exp(theta_eps_s)
        
        ## then
        spec_dens_y_tf <- spec_dens_x_tf + spec_dens_eps_tf
        
        log_likelihood_tf <- - tf$math$log(spec_dens_y_tf) - tf$multiply(I_i, tf$math$reciprocal(spec_dens_y_tf))
        
      })
      grad_tf %<-% tape1$gradient(log_likelihood_tf, c(theta_phi_s, theta_eta_s, theta_eps_s))
      
      grad_tf <- tf$reshape(grad_tf, c(3L, dim(grad_tf[[1]])))
    })
    
    grad2_tf %<-% tape2$jacobian(grad_tf, c(theta_phi_s, theta_eta_s, theta_eps_s))
    
    return(list(log_likelihood = log_likelihood_tf,
                grad = grad_tf,
                hessian = grad2_tf))
  }
)

compute_grad_arctanh_test <- tf_function(
  testf <- function(samples_tf, I_i, freq_i) {
    log_likelihood_tf <- 0
    with (tf$GradientTape() %as% tape2, {
      with (tf$GradientTape(persistent = TRUE) %as% tape1, {
        
        phi_s <- tf$math$tanh(samples_tf[, 1])
        spec_dens_x_tf <- tf$math$divide(tf$math$exp(samples_tf[, 2]), 1 + tf$math$square(phi_s) -
                                           tf$math$multiply(2, tf$math$multiply(phi_s, tf$math$cos(freq_i))))
        
        ## add spec_dens_eps here
        spec_dens_eps_tf <- tf$math$exp(samples_tf[, 3])
        
        ## then
        spec_dens_y_tf <- spec_dens_x_tf + spec_dens_eps_tf
        
        log_likelihood_tf <- - tf$math$log(spec_dens_y_tf) - tf$multiply(I_i, tf$math$reciprocal(spec_dens_y_tf))
        
      })
      # vars <- tf$reshape(cbind(theta_phi_s, theta_eta_s, theta_eps_s), c(length(theta_phi_s), 3L))
      grad_tf %<-% tape1$gradient(log_likelihood_tf, samples_tf)
      # grad_tf %<-% tape1$gradient(log_likelihood_tf, c(theta_phi_s, theta_eta_s, theta_eps_s))
      
      # grad_tf <- tf$reshape(tf$transpose(grad_tf), c(dim(grad_tf[[1]]), 3L))
    })
    # grad2_tf %<-% tape2$jacobian(grad_tf, c(theta_phi_s, theta_eta_s, theta_eps_s))
    
    # vars <- tf$reshape(c(theta_phi_s, theta_eta_s, theta_eps_s), dim(grad_tf))
    grad2_tf %<-% tape2$batch_jacobian(grad_tf, samples_tf)
    # grad2_tf %<-% tape2$batch_jacobian(grad_tf, vars)
    
    E_grad_tf <- tf$reduce_mean(grad_tf, 0L)
    E_hessian_tf <- tf$reduce_mean(grad2_tf, 0L)
    
    return(list(log_likelihood = log_likelihood_tf,
                grad = grad_tf,
                hessian = grad2_tf,
                E_grad = E_grad_tf,
                E_hessian = E_hessian_tf))
  }
)