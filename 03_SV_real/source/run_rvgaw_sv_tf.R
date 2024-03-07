run_rvgaw_sv <- function(y, phi = NULL, sigma_eta = NULL, sigma_xi = NULL, 
                         prior_mean = 0, prior_var = 1, 
                         deriv = "tf", S = 500,
                         use_tempering = T, temper_first = T, 
                         temper_schedule = rep(1/10, 10),
                         n_temper = 100,
                         reorder = "random", reorder_seed = 2023,
                         transform = "logit",
                         use_welch = F) {
  
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
  
  if (use_welch) {
    par(mfrow = c(1,1))
    test1 <- empiricalSpectrum(as.ts(y_tilde))
    test2 <- welchPSD(as.ts(y_tilde), seglength = length(y_tilde)/10, windowingPsdCorrection = F)
    # plot(test1$frequency, test1$power, col="black", type = "l")
    # lines(test2$frequency, test2$power, col="red", type = "l", lwd = 2)
    
    freq2 <- test2$frequency[2:length(test2$frequency)]
    I2 <- test2$power[2:length(test2$power)]/2
    plot(freq, I, type = "l") # original freq and periodogram
    lines(freq2, I2, col = "red", lwd = 2) # welch's
    
    freq <- freq2
    I <- I2
    browser()  
    
    # ## Try Welch's method here?
    # n_segments <- 10
    # seg_length <- n/n_segments
    # # split the vector by length 
    # seg_ind <- split(1:n,ceiling(seq_along(1:n) / seg_length))
    # seg_period <- list()
    # for (q in 1:n_segments) {
    #   fft_output <- fft(y_tilde[seg_ind[[q]]])
    #   seg_period[[q]] <- 1/seg_length * Mod(fft_output)^2
    # }
    
    # avg_periodogram <- 1/n_segments * Reduce("+", seg_period)
    # avg_periodogram_half <- avg_periodogram[1:floor((seg_length-1)/2)]
    
    # I <- avg_periodogram_half
    # freq <- 2 * pi * 1:floor((seg_length-1)/2) / seg_length
    # browser()
  }
  
  
  if (reorder == "decreasing") {
    sorted_freq <- sort(freq, decreasing = T, index.return = T)
    indices <- sorted_freq$ix
    reordered_freq <- sorted_freq$x
    reordered_I <- I[indices]
  } else if (reorder == "random") {
    set.seed(reorder_seed)
    indices <- sample(1:length(freq), length(freq))
    reordered_freq <- freq[indices]
    reordered_I <- I[indices]
  } else if (reorder > 0) {
    n_reorder <- reorder
    # n_others <- length(freq - n_reorder)
    original <- (n_reorder+1):length(freq)
    inds <- ceiling(seq(n_reorder, length(freq) - n_reorder, length.out = n_reorder))
    new <- original
    for (j in 1:n_reorder) {
      new <- append(new, j, after = inds[j])
    }
    
    reordered_freq <- freq[new]
    reordered_I <- I[new]
  } else { ## do nothing
    reordered_freq <- freq
    reordered_I <- I
  }
  
  # plot(reordered_freq, type = "l")
  # lines(freq, col = "red")
  freq <- reordered_freq
  I <- reordered_I 
  # browser()
  
  if (use_tempering) {
    if (temper_first) {
      cat("Damping the first ", n_temper, "frequencies... \n")
    } else {
      cat("Damping the last ", n_temper, "frequencies... \n")
    }
  }
  
  for (i in 1:length(freq)) {
    
    # cat("i =", i, "\n")
    
    a_vals <- 1
    if (use_tempering) {
      if (temper_first) {
        
        if (i <= n_temper) { # only temper the first n_temper observations
          a_vals <- temper_schedule
        }  
      } else {
        if (i > length(freq) - n_temper) { # only temper the first n_temper observations
          
          a_vals <- temper_schedule
        }
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
        
        grad_expr <- Deriv(log_likelihood_fun, c("theta_phi", "theta_eta"))
        grad2_expr <- Deriv(grad_expr, c("theta_phi", "theta_eta"))
        
        # Gradient
        grad_deriv <- mapply(grad_expr, theta_phi = theta_phi, 
                             theta_eta = theta_eta, theta_xi = theta_xi,
                             omega_k = freq[i], I_k = I[[i]],
                             transform = transform)
        
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
        
        if (transform == "arctanh") {
          tf_out <- compute_grad_arctanh(samples_tf, I_i_tf, freq_i_tf)
        } else { # use logit
          tf_out <- compute_grad_logit(samples_tf, I_i_tf, freq_i_tf)
        }
        
        # browser()
        
        grads_tf <- tf_out$grad
        hessians_tf <- tf_out$hessian
        E_grad_tf <- tf$reduce_mean(grads_tf, 0L)
        E_hessian_tf <- tf$reduce_mean(hessians_tf, 0L)
        
        tf.t2 <- proc.time()
        
        E_grad <- as.vector(E_grad_tf)
        E_hessian <- as.matrix(E_hessian_tf)
        
      }
      prec_temp <- prec_temp - a * E_hessian
      
      if(any(eigen(prec_temp)$value < 0)) {
        prec_temp <- prec_temp + diag(1, nrow(prec_temp))
        
      }
      
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
  
  if (transform == "arctanh") {
    rvgaw.post_samples_phi <- tanh(theta.post_samples[, 1])
  } else {
    rvgaw.post_samples_phi <- 1 / (1 + exp(-theta.post_samples[, 1]))
  }
  
  rvgaw.post_samples_eta <- sqrt(exp(theta.post_samples[, 2]))
  
  rvgaw.post_samples <- list(phi = rvgaw.post_samples_phi,
                             sigma_eta = rvgaw.post_samples_eta) #,
  
  ## Save results
  rvgaw_results <- list(mu = rvgaw.mu_vals,
                        prec = rvgaw.prec,
                        post_samples = rvgaw.post_samples,
                        transform = transform,
                        S = S,
                        use_tempering = use_tempering,
                        temper_first = temper_first,
                        temper_schedule = a_vals,
                        time_elapsed = rvgaw.t2 - rvgaw.t1)
  
  return(rvgaw_results)
}

log_likelihood_fun <- function(theta_phi, theta_eta, theta_xi, transform,
                               omega_k, I_k) { 
  
  phi <- 0
  if (transform == "arctanh") {
    phi <- tanh(theta_phi)
  } else {
    phi <- 1 / (1 + exp(-theta_phi))
  }
  
  sigma_eta <- sqrt(exp(theta_eta))
  sigma_xi <- sqrt(exp(theta_xi))
  
  spec_dens_x <- sigma_eta^2/( 1 + phi^2 - 2 * phi * cos(omega_k))
  spec_dens_xi <- sigma_xi^2 #/ (2*pi)
  spec_dens_y <- spec_dens_x + spec_dens_xi
  llh <- - log(spec_dens_y) - I_k / spec_dens_y
}

# compute_grad_arctanh <- tf_function(
#   testf <- function(theta_phi, theta_eta, theta_xi, I_i, freq_i) {
#     with (tf$GradientTape() %as% tape2, {
#       with (tf$GradientTape(persistent = TRUE) %as% tape1, {
#         
#         phi <- tf$math$tanh(theta_phi)
#         spec_dens_x_tf <- tf$math$divide(tf$math$exp(theta_eta), 1 + tf$math$square(phi) -
#                                            tf$math$multiply(2, tf$math$multiply(phi, tf$math$cos(freq_i))))
#         
#         ## add spec_dens_xi here
#         spec_dens_xi_tf <- tf$math$exp(theta_xi)
#         
#         ## then
#         spec_dens_y_tf <- spec_dens_x_tf + spec_dens_xi_tf
#         
#         log_likelihood_tf <- - tf$math$log(spec_dens_y_tf) - tf$multiply(I_i, tf$math$reciprocal(spec_dens_y_tf))
#         
#       })
#       grad_tf %<-% tape1$gradient(log_likelihood_tf, c(theta_phi, theta_eta))
#       
#       grad_tf <- tf$reshape(grad_tf, c(2L, dim(grad_tf[[1]])))
#     })
#     grad2_tf %<-% tape2$jacobian(grad_tf, c(theta_phi, theta_eta))
#     
#     return(list(llh = log_likelihood_tf, 
#                 grad = grad_tf,
#                 hessian = grad2_tf))
#   }
# )


compute_grad_arctanh <- tf_function(
  compute_grad_arctanh <- function(samples_tf, I_i, freq_i) {
    with (tf$GradientTape() %as% tape2, {
      with (tf$GradientTape(persistent = TRUE) %as% tape1, {
        
        theta_phi <- samples_tf[, 1]
        theta_eta <- samples_tf[, 2]
        
        theta_xi <- tf$constant(rep(log(pi^2/2), as.integer(length(theta_phi))))
        
        phi <- tf$math$tanh(theta_phi)
        spec_dens_x_tf <- tf$math$divide(tf$math$exp(theta_eta), 1 + tf$math$square(phi) -
                                           tf$math$multiply(2, tf$math$multiply(phi, tf$math$cos(freq_i))))
        
        ## add spec_dens_xi here
        spec_dens_xi_tf <- tf$math$exp(theta_xi)
        
        ## then
        spec_dens_y_tf <- spec_dens_x_tf + spec_dens_xi_tf
        
        log_likelihood_tf <- - tf$math$log(spec_dens_y_tf) - tf$multiply(I_i, tf$math$reciprocal(spec_dens_y_tf))
        
      })
      grad_tf %<-% tape1$gradient(log_likelihood_tf, samples_tf)
      
    })
    grad2_tf %<-% tape2$batch_jacobian(grad_tf, samples_tf)
    
    return(list(llh = log_likelihood_tf, 
                grad = grad_tf,
                hessian = grad2_tf))
  }
)

compute_grad_logit <- tf_function(
  compute_grad_logit <- function(samples_tf, I_i, freq_i) {
    with (tf$GradientTape() %as% tape2, {
      with (tf$GradientTape(persistent = TRUE) %as% tape1, {
        
        theta_phi <- samples_tf[, 1]
        theta_eta <- samples_tf[, 2]
        
        theta_xi <- tf$constant(rep(log(pi^2/2), as.integer(length(theta_phi))))
        
        phi <- tf$math$divide(1, 1 + tf$math$exp(-theta_phi))
        spec_dens_x_tf <- tf$math$divide(tf$math$exp(theta_eta), 1 + tf$math$square(phi) -
                                           tf$math$multiply(2, tf$math$multiply(phi, tf$math$cos(freq_i))))
        
        ## add spec_dens_xi here
        spec_dens_xi_tf <- tf$math$exp(theta_xi)
        
        ## then
        spec_dens_y_tf <- spec_dens_x_tf + spec_dens_xi_tf
        
        log_likelihood_tf <- - tf$math$log(spec_dens_y_tf) - tf$multiply(I_i, tf$math$reciprocal(spec_dens_y_tf))
        
      })
      grad_tf %<-% tape1$gradient(log_likelihood_tf, samples_tf)
      
    })
    grad2_tf %<-% tape2$batch_jacobian(grad_tf, samples_tf)
    
    return(list(llh = log_likelihood_tf, 
                grad = grad_tf,
                hessian = grad2_tf))
  }
)