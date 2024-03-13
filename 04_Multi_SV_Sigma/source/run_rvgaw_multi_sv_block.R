run_rvgaw_multi_sv <- function(data, prior_mean, prior_var, S,
                               use_tempering = T, temper_first = T,
                               temper_schedule, 
                               reorder,
                               reorder_seed = 2023, use_cholesky = F,
                               transform = "arctanh",
                               n_post_samples = 10000,
                               use_median = F, use_welch = F) {
  rvgaw.t1 <- proc.time()
  
  Y <- data
  d <- ncol(Y)
  Tfin <- nrow(Y)
  
  if (use_cholesky) {
    param_dim <- d + (d*(d-1)/2 + d) # m^2 AR parameters, 
    # m*(m-1)/2 + m parameters from the lower Cholesky factor of Sigma_eta
  } else {
    param_dim <- d + d
  }
  
  rvgaw.mu_vals <- list()
  rvgaw.mu_vals[[1]] <- prior_mean
  
  rvgaw.prec <- list()
  rvgaw.prec[[1]] <- chol2inv(chol(prior_var))
  
  # ## Calculation of Whittle likelihood
  pgram_out <- compute_periodogram(Y)
  freq <- pgram_out$freq
  I <- pgram_out$periodogram
  
  # test <- sapply(1:dim(I)[3], function(x) Mod(I[,,x][1,1])) # extract periodogram elements
  # test22 <- sapply(1:dim(I)[3], function(x) I[,,x][2,2]) # extract periodogram elements
  # test21 <- sapply(1:dim(I)[3], function(x) I[,,x][2,2]) # extract periodogram elements
  
  ## Reorder frequencies if needed
  if (reorder == "decreasing") {
    sorted_freq <- sort(freq, decreasing = T, index.return = T)
    indices <- sorted_freq$ix
    reordered_freq <- sorted_freq$x
    reordered_I <- I[,,indices]
  } else if (reorder == "random") {
    set.seed(reorder_seed)
    indices <- sample(1:length(freq), length(freq))
    reordered_freq <- freq[indices]
    reordered_I <- I[,,indices]
  } else if (reorder > 0) {
    n_reorder <- reorder
    original <- (n_reorder+1):length(freq)
    inds <- ceiling(seq(n_reorder, length(freq) - n_reorder, length.out = n_reorder))
    new <- original
    for (j in 1:n_reorder) {
      new <- append(new, j, after = inds[j])
    }
    
    reordered_freq <- freq[new]
    reordered_I <- I[,,new]
  } else { ## do nothing
    reordered_freq <- freq
    reordered_I <- I
  }
  
  freq <- reordered_freq
  I <- reordered_I 
  
  all_blocks <- as.list(1:length(freq))
  
  if (!is.null(nblocks)) {
    # Split frequencies into blocks
    # Last block may not have the same size as the rest
    # if the number of frequencies to be divided into blocks
    # is not divisible by the number of blocks
    indiv <- list()
    vec <- c()
    if (reorder == 0) { # leave the first n_indiv frequencies alone, cut the rest into blocks
      indiv <- as.list(1:n_indiv)
      vec <- (n_indiv+1):length(freq)
      blocks <- split(vec, cut(seq_along(vec), nblocks, labels = FALSE))
      all_blocks <- c(indiv, blocks) 
    } else if (reorder == "decreasing") { # leave the last n_indiv frequencies alone, cut the rest into blocks
      indiv <- as.list((length(freq) - n_indiv):length(freq))
      vec <- 1:(length(freq) - n_indiv)
      blocks <- split(vec, cut(seq_along(vec), nblocks, labels = FALSE))
      all_blocks <- c(blocks, indiv)
    }
  }
  
  # browser()
  if (use_tempering) {
    if (temper_first) {
      cat("Damping the first ", n_temper, "frequencies... \n")
    } else {
      cat("Damping the last ", n_temper, "frequencies... \n")
    }
  }
  
  n_updates <- length(all_blocks)
  for (i in 1:n_updates) {
    cat("i =", i, "\n")
    blockinds <- all_blocks[[i]]
    
    # if (i == 106) {
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
    
    E_grad <- 0
    E_hessian <- 0
    
    for (v in 1:length(a_vals)) { # for each step in the tempering schedule
      
      a <- a_vals[v]
      
      P <- chol2inv(chol(prec_temp))
      samples <- rmvnorm(S, mu_temp, P)
      
      samples_tf <- tf$Variable(samples)
     
      I_tf <- tf$Variable(I[,,blockinds])
      freq_tf <- tf$Variable(freq[blockinds])
     
      tf_out <- compute_grad_hessian(samples_tf, I_i = I_tf, freq_i = freq_tf,
                                      blocksize = length(blockinds),
                                     use_cholesky = use_cholesky, 
                                     transform = transform)
         
      grads_tf <- tf_out$grad
      hessians_tf <- tf_out$hessian
      
      E_grad_tf <- 0
      E_hessian_tf <- 0
      if (use_median) {
        E_grad_tf <- tfp$stats$percentile(grads_tf, 50, 0L)
        E_hessian_tf <- tfp$stats$percentile(hessians_tf, 50, 0L)
      } else {
        E_grad_tf <- tf$reduce_mean(grads_tf, 0L)
        E_hessian_tf <- tf$reduce_mean(hessians_tf, 0L)
      }
      E_grad <- as.vector(E_grad_tf)
      E_hessian <- as.matrix(E_hessian_tf)
      
      prec_temp <- prec_temp - a * E_hessian
      
      # prec_temp <- approx_hessian(prec_temp, method = "multiple_id")
      eigvals <- eigen(prec_temp)$value
      if(any(eigvals < 0)) {
        # browser() ## try nearPD() funciton from the Matrix package here
        neg_eigval <- eigvals[eigvals < 0]
        cat("Warning: precision matrix has negative eigenvalue", neg_eigval, "\n")
        # prec_temp <- as.matrix(nearPD(prec_temp)$mat)
      }
      
      mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * E_grad)
    }  
    
    rvgaw.prec[[i+1]] <- prec_temp
    rvgaw.mu_vals[[i+1]] <- mu_temp
    
    if (i %% floor(n_updates+1/10) == 0) {
      cat(floor(i/n_updates+1 * 100), "% complete \n")
    }
    
  }
  rvgaw.t2 <- proc.time()
  
  ## Posterior samples
  rvgaw.post_var <- chol2inv(chol(rvgaw.prec[[n_updates+1]]))
  
  rvgaw.post_samples <- rmvnorm(n_post_samples, rvgaw.mu_vals[[n_updates+1]], rvgaw.post_var) # these are samples of beta, log(sigma_a^2), log(sigma_e^2)
  
  ## Construct A and Sigma_eta from these posterior samples
  ## Maybe this should be done in tensorflow?
  rvgaw.post_samples2 <- lapply(seq_len(nrow(rvgaw.post_samples)), function(i) rvgaw.post_samples[i,])
  
  ### the first 4 elements will be used to construct A
  # post_samples_A <- lapply(rvgaw.post_samples2, function(x) matrix(x[1:(d^2)], d, d, byrow = T))
  if (transform == "arctanh") {
    post_samples_Phi <- lapply(rvgaw.post_samples2, function(x) diag(tanh(x[1:d])))
  } else {
    post_samples_Phi <- lapply(rvgaw.post_samples2, function(x) diag(1/(1+exp(-x[1:d]))))
  }
  
  if (use_cholesky) {
    post_samples_Sigma_eta <- lapply(rvgaw.post_samples2, construct_Sigma_eta, d = d, use_chol = use_cholesky)
  } else {
    post_samples_Sigma_eta <- lapply(rvgaw.post_samples2, function(x) diag(exp(x[(d^2+1):param_dim])))
  }
  ## Transform samples of A into samples of Phi via the mapping in Ansley and Kohn (1986)
  # post_samples_Phi <- mapply(backward_map, post_samples_A, post_samples_Sigma_eta, SIMPLIFY = F)
  rvgaw.post_samples <- list(Phi = post_samples_Phi,
                             Sigma_eta = post_samples_Sigma_eta)
  
  ## Save results
  rvgaw_results <- list(data = Y, 
                        mu = rvgaw.mu_vals,
                        prec = rvgaw.prec,
                        prior_mean = prior_mean,
                        prior_var = prior_var,
                        post_samples = rvgaw.post_samples,
                        transform = transform,
                        S = S,
                        use_tempering = use_tempering,
                        temper_first = temper_first,
                        temper_schedule = a_vals,
                        time_elapsed = rvgaw.t2 - rvgaw.t1)
  
}

# ### the last 3 will be used to construct L
# construct_Sigma_eta <- function(theta, d, use_chol) { #d is the dimension of Sigma_eta
#   nlower <- d*(d-1)/2
#   L <- diag(exp(theta[(d+1):(2*d)]))
#   offdiags <- theta[-(1:(2*d))] # off diagonal elements are those after the first 2*d elements
#   
#   if (use_chol) {
#     for (k in 1:nlower) {
#       ind <- index_to_i_j_rowwise_nodiag(k)
#       L[ind[1], ind[2]] <- offdiags[k]
#     }
#     Sigma_eta <- L %*% t(L)
#   } else {
#     Sigma_eta <- L
#   }
#   return(Sigma_eta)
# }

approx_hessian <- function(H, method = "flip_neg_eigvals", beta = 0.001) { # or method = "flip_neg_eigvals"
  eigs = eigen(H)$values #np.linalg.eigvals(H)
  
  Hbar <- H
  if (min(eigs) < 0) { # then use one of two methods to modify H so that it's posdef
    cat("Warning: precision matrix has negative eigenvalues", eigs[eigs < 0], "\n")
    if (method == "flip_neg_eigvals") { # flip negative eigvals to positive 
      print("Flipping eigenvalues...")
      mod_mat <- H > 0
      mod_mat[mod_mat==0] = -1
      Hbar <- H * mod_mat
      browser()
    } else if (method == "multiple_id") { # add a multiple of the identity matrix to the hessian
      print("Inflating diagonal entries of the precision matrix...")
      tau <- - min(eigs) + beta # beta is the new "minimum eigenvalue"
      Hbar <- H + diag(nrow(H)) * tau # modifies using multiples of identity
    }
  }
  
  return(Hbar)
}

calc_periodogram <- function(M) {
  crosspd <- M %*% t(Conj(M))
  return(crosspd)
}