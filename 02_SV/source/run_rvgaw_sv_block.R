run_rvgaw_sv <- function(y, phi = NULL, sigma_eta = NULL, sigma_xi = NULL, 
                         prior_mean = 0, prior_var = 1, 
                         deriv = "tf", n_post_samples = 10000,
                         S = 500,
                         use_tempering = T, temper_first = T, 
                         temper_schedule = rep(1/10, 10),
                         n_temper = 100,
                         reorder = "random", reorder_seed = 2023,
                         transform = "arctanh",
                         nblocks = NULL, n_indiv = NULL) {
  
  print("Starting R-VGAL with Whittle likelihood...")
  
  rvgaw.t1 <- proc.time()
  
  n <- length(y)
  
  rvgaw.mu_vals <- list()
  rvgaw.mu_vals[[1]] <- prior_mean
  
  rvgaw.prec <- list()
  rvgaw.prec[[1]] <- chol2inv(chol(prior_var))
  
  pgram_out <- compute_periodogram(y)
  freq <- pgram_out$freq
  I <- pgram_out$periodogram
  
  
  ## Reorder frequencies if needed
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
  
  all_blocks <- as.list(1:length(freq))
  
  if (!is.null(nblocks)) {
    # Split frequencies into blocks
    # Last block may not have the same size as the rest
    # if the number of frequencies to be divided into blocks
    # is not divisible by the number of blocks
    indiv <- list()
    vec <- c()
    if (reorder == 0) { # leave the first n_indiv frequencies alone, cut the rest into blocks
      vec <- (n_indiv+1):length(freq)
      blocks <- split(vec, cut(seq_along(vec), nblocks, labels = FALSE))
      if (n_indiv == 0) {
        all_blocks <- blocks
      } else {
        indiv <- as.list(1:n_indiv)
        all_blocks <- c(indiv, blocks)    
      }
    } else if (reorder == "decreasing") { # leave the last n_indiv frequencies alone, cut the rest into blocks
      indiv <- as.list((length(freq) - n_indiv):length(freq))
      vec <- 1:(length(freq) - n_indiv)
      blocks <- split(vec, cut(seq_along(vec), nblocks, labels = FALSE))
      all_blocks <- c(blocks, indiv)
    }
  }
  
  n_updates <- length(all_blocks)
  for (i in 1:n_updates) {
    cat("i =", i, "\n")
    blockinds <- all_blocks[[i]]
    
    # if (i == 126) {
    #   browser()
    # }
    
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
        
        tf.t1 <- proc.time()
        
        theta_phi_tf <- tf$Variable(theta_phi)
        theta_eta_tf <- tf$Variable(theta_eta)
        theta_xi_tf <- tf$Variable(rep(theta_xi, S))#tf$Variable(theta_xi)
        
        samples_tf <- tf$Variable(samples, dtype = "float32")
        
        freq_i_tf <- tf$constant(freq[blockinds])
        I_i_tf <- tf$constant(I[blockinds])
        
        if (transform == "arctanh") {
          # tf_out <- compute_grad_arctanh(samples_tf, I_i_tf, freq_i_tf)
          tf_out <- compute_grad_arctanh_block(samples_tf, I_i_tf, freq_i_tf,
                                               blocksize = length(blockinds))
          
        } else { # use logit
          tf_out <- compute_grad_logit(samples_tf, I_i_tf, freq_i_tf)
        }
        
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
    
    if (i %% floor(n_updates/10) == 0) {
      cat(floor(i/n_updates * 100), "% complete \n")
    }
    
  }
  
  rvgaw.t2 <- proc.time()
  
  ## Posterior samples
  rvgaw.post_var <- chol2inv(chol(rvgaw.prec[[n_updates+1]]))
  
  theta.post_samples <- rmvnorm(10000, rvgaw.mu_vals[[n_updates+1]], rvgaw.post_var) # these are samples of beta, log(sigma_a^2), log(sigma_e^2)
  
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

compute_grad_arctanh_block <- tf_function(
  compute_grad_arctanh_block <- function(samples_tf, I_i, freq_i, blocksize) {
    log_likelihood_tf <- 0
    with(tf$GradientTape() %as% tape2, {
      with(tf$GradientTape(persistent = TRUE) %as% tape1, {
        
        S <- as.integer(nrow(samples_tf))
        
        # nfreq <- as.integer(length(freq_i))
        phi_s <- tf$math$tanh(samples_tf[, 1])
        phi_s <- tf$reshape(phi_s, c(length(phi_s), 1L, 1L)) # S x 1 x 1
        freq_i <- tf$reshape(freq_i, c(1L, blocksize, 1L)) # 1 x blocksize x 1
        
        sigma_eta2_s <- tf$math$exp(samples_tf[, 2])
        sigma_eta2_s <- tf$reshape(sigma_eta2_s, c(dim(sigma_eta2_s), 1L, 1L))
        sigma_eta2_tiled <- tf$tile(sigma_eta2_s, c(1L, blocksize, 1L))
        
        spec_dens_x_tf <- tf$math$divide(sigma_eta2_tiled, 
                                         1 + tf$tile(tf$math$square(phi_s), c(1L, blocksize, 1L)) -
                                           tf$math$multiply(2, tf$math$multiply(phi_s, tf$math$cos(freq_i))))
        
        ## add spec_dens_eps here
        theta_xi <- tf$constant(pi^2/2, shape = c(1L, 1L, 1L))
        spec_dens_xi_tf <- tf$tile(theta_xi, c(S, blocksize, 1L))
        
        # spec_dens_eps_tf <- tf$math$exp(samples_tf[, 3])
        # spec_dens_eps_tf <- tf$reshape(spec_dens_eps_tf, c(S, 1L, 1L))
        ## then
        spec_dens_y_tf <- spec_dens_x_tf + spec_dens_xi_tf #tf$tile(spec_dens_eps_tf, c(1L, nfreq, 1L))
        
        I_i <- tf$reshape(I_i, c(1L, blocksize, 1L))
        I_tile <- tf$tile(I_i, c(S, 1L, 1L))
        log_likelihood_tf <- -tf$math$log(spec_dens_y_tf) - tf$multiply(I_i, tf$math$reciprocal(spec_dens_y_tf))
        
        log_likelihood_tf <- tf$math$reduce_sum(log_likelihood_tf, 1L) # sum all log likelihoods over the block
      })
      grad_tf %<-% tape1$gradient(log_likelihood_tf, samples_tf)
      # grad_tf <- tf$reshape(tf$transpose(grad_tf), c(dim(grad_tf[[1]]), 3L))
    })
    
    grad2_tf %<-% tape2$batch_jacobian(grad_tf, samples_tf)
    # grad2_tf %<-% tape2$batch_jacobian(grad_tf, vars)
    
    E_grad_tf <- tf$reduce_mean(grad_tf, 0L)
    E_hessian_tf <- tf$reduce_mean(grad2_tf, 0L)
    
    return(list(
      log_likelihood = log_likelihood_tf,
      grad = grad_tf,
      hessian = grad2_tf,
      E_grad = E_grad_tf,
      E_hessian = E_hessian_tf
    ))
  }
)


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