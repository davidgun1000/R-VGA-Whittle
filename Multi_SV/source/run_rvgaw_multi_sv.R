run_rvgaw_multi_sv <- function(data, prior_mean, prior_var, S,
                               use_tempering = T, temper_schedule, 
                               reorder_freq = T, decreasing = T,
                               reorder_seed = 2023, use_cholesky = F,
                               n_post_samples = 10000) {
  rvgaw.t1 <- proc.time()
  
  d <- nrow(data)
  if (use_cholesky) {
    param_dim <- d^2 + (d*(d-1)/2 + d) # m^2 AR parameters, 
    # m*(m-1)/2 + m parameters from the lower Cholesky factor of Sigma_eta
  } else {
    param_dim <- d^2 + d
  }
  
  rvgaw.mu_vals <- list()
  rvgaw.mu_vals[[1]] <- prior_mean
  
  rvgaw.prec <- list()
  rvgaw.prec[[1]] <- chol2inv(chol(prior_var))
  
  ## Generate a bunch of samples from the initial distribution
  ## Initial values: sample params from prior
  samples <- rmvnorm(S, prior_mean, prior_var)
  
  ## Construct A and Sigma_eta from these prior samples
  ## Maybe this should be done in tensorflow?
  samples2 <- lapply(seq_len(nrow(samples)), function(i) samples[i,])

  ### the first 4 elements will be used to construct A
  A_samples <- lapply(samples2, function(x) matrix(x[1:(d^2)], d, d, byrow = T))

  ### the last 3 will be used to construct L
  construct_Sigma_eta <- function(theta) {
    L <- diag(exp(theta[(d^2+1):param_dim]))
    # L[2,1] <- theta[7]
    # Sigma_eta <- L %*% t(L)
    Sigma_eta <- L
    return(Sigma_eta)
  }

  # L <- diag(exp(theta_samples[1, 5:6]))
  # L[2,1] <- theta_samples[1, 7]
  # Sigma_eta_curr <- L %*% t(L)

  Sigma_eta_samples <- lapply(samples2, construct_Sigma_eta)

  ## Transform samples of A into samples of Phi via the mapping in Ansley and Kohn (1986)
  Phi_samples <- mapply(backward_map, A_samples, Sigma_eta_samples, SIMPLIFY = F)

  # j <- 2
  # llh <- list()
  # for (s in 1:S) {
  #   params_s <- list(Phi = Phi_samples[[s]], Sigma_eta = Sigma_eta_samples[[s]])
  #   llh[[s]] <- compute_partial_whittle_likelihood(Y = Y, params = params_s, j = j)
  # }
  
  # ## Calculation of Whittle likelihood
  ## Fourier frequencies
  k <- seq(-ceiling(Tfin/2)+1, floor(Tfin/2), 1)
  k_in_likelihood <- k [k >= 1 & k <= floor((Tfin-1)/2)]
  freq <- 2 * pi * k_in_likelihood / Tfin
  
  # ## astsa package
  Z <- log(Y^2) - rowMeans(log(Y^2))
  fft_out <- mvspec(t(Z), detrend = F, plot = F)
  I <- fft_out$fxx
  
  if (reorder_freq) { # randomise order of frequencies and periodogram components
    
    if (decreasing) {
      sorted_freq <- sort(freq, decreasing = T, index.return = T)
      indices <- sorted_freq$ix
      reordered_freq <- sorted_freq$x
      reordered_I <- I[,,indices]
    } else {
      set.seed(reorder_seed)
      indices <- sample(1:length(freq), length(freq))
      reordered_freq <- freq[indices]
      reordered_I <- I[,,indices]
    }
    
    # plot(reordered_freq, type = "l")
    # lines(freq, col = "red")
    freq <- reordered_freq
    I <- reordered_I 
  }
  
  #### TF starts ##########
  # j <- 1
  # samples_tf <- tf$Variable(samples)
  # I_tf <- tf$Variable(I_all[,,j])
  # freq_tf <- tf$Variable(freq[j])
  # 
  # llh_test <- compute_grad_hessian(samples_tf, I_i = I_tf, freq_i = freq_tf)
  
  # ### the last 3 will be used to construct L
  # construct_Sigma_eta <- function(theta) {
  #   L <- diag(exp(theta[5:6]))
  #   L[2,1] <- theta[7]
  #   Sigma_eta <- L %*% t(L)
  #   return(Sigma_eta)
  # }
  
  # Find the expected grad and expected Hessian
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
    
    # grads <- list()
    # hessian <- list()
    
    E_grad <- 0
    E_hessian <- 0
    
    for (v in 1:length(a_vals)) { # for each step in the tempering schedule
      
      a <- a_vals[v]
      
      P <- chol2inv(chol(prec_temp))
      samples <- rmvnorm(S, mu_temp, P)
      
      # theta_phi <- samples[, 1]
      # theta_eta <- samples[, 2]
      # # theta_xi <- samples[, 3]
      # theta_xi <- log(pi^2/2)
      
      samples_tf <- tf$Variable(samples)
      I_tf <- tf$Variable(I[,,i])
      freq_tf <- tf$Variable(freq[i])
      
      tf_out <- compute_grad_hessian(samples_tf, I_i = I_tf, freq_i = freq_tf)
      
      # llh <- list()
      # for (s in 1:S) {
      # 
      #   ### the first 4 elements will be used to construct A
      #   A_s <- matrix(samples[s, 1:4], 2, 2, byrow = T)
      # 
      #   Sigma_eta_s <- diag(exp(samples[s, 5:6]))
      # 
      #   ## Transform samples of A into samples of Phi via the mapping in Ansley and Kohn (1986)
      #   Phi_s <- backward_map(A_s, Sigma_eta_s)
      # 
      #   llh[[s]] <- compute_partial_whittle_likelihood(Y, params = list(Phi = Phi_s,
      #                                                                   Sigma_eta = Sigma_eta_s),
      #                                                  j = indices[i])
      # }
      
      grads_tf <- tf_out$grad
      hessians_tf <- tf_out$hessian
      E_grad_tf <- tf$reduce_mean(grads_tf, 0L)
      E_hessian_tf <- tf$reduce_mean(hessians_tf, 0L)
      
      E_grad <- as.vector(E_grad_tf)
      E_hessian <- as.matrix(E_hessian_tf)
      
      prec_temp <- prec_temp - a * E_hessian
      
      eigvals <- eigen(prec_temp)$value
      if(any(eigvals < 0)) {
        # browser() ## try nearPD() funciton from the Matrix package here
        neg_eigval <- eigvals[eigvals < 0]
        cat("Warning: precision matrix has negative eigenvalue", neg_eigval, "\n")
        prec_temp <- as.matrix(nearPD(prec_temp)$mat)
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
  
  rvgaw.post_samples <- rmvnorm(n_post_samples, rvgaw.mu_vals[[length(freq)]], rvgaw.post_var) # these are samples of beta, log(sigma_a^2), log(sigma_e^2)
  
  ## Construct A and Sigma_eta from these posterior samples
  ## Maybe this should be done in tensorflow?
  rvgaw.post_samples2 <- lapply(seq_len(nrow(rvgaw.post_samples)), function(i) rvgaw.post_samples[i,])
  
  ### the first 4 elements will be used to construct A
  post_samples_A <- lapply(rvgaw.post_samples2, function(x) matrix(x[1:(d^2)], d, d, byrow = T))
  
  if (use_cholesky) {
    ### the last 3 will be used to construct L
    construct_Sigma_eta <- function(theta) {
      L <- diag(exp(theta[5:6]))
      L[2,1] <- theta[7]
      Sigma_eta <- L %*% t(L)
      return(Sigma_eta)
    }
    post_samples_Sigma_eta <- lapply(rvgaw.post_samples2, construct_Sigma_eta)
  } else {
    post_samples_Sigma_eta <- lapply(rvgaw.post_samples2, function(x) diag(exp(x[(d^2+1):param_dim])))
  }
  
  # L <- diag(exp(theta_samples[1, 5:6]))
  # L[2,1] <- theta_samples[1, 7]
  # Sigma_eta_curr <- L %*% t(L)
  
  
  ## Transform samples of A into samples of Phi via the mapping in Ansley and Kohn (1986)
  post_samples_Phi <- mapply(backward_map, post_samples_A, post_samples_Sigma_eta, SIMPLIFY = F)
  
  rvgaw.post_samples <- list(Phi = post_samples_Phi,
                             Sigma_eta = post_samples_Sigma_eta)
  
  ## Save results
  rvgaw_results <- list(mu = rvgaw.mu_vals,
                        prec = rvgaw.prec,
                        prior_mean = prior_mean,
                        prior_var = prior_var,
                        post_samples = rvgaw.post_samples,
                        S = S,
                        use_tempering = use_tempering,
                        temper_schedule = a_vals,
                        time_elapsed = rvgaw.t2 - rvgaw.t1)
  
}