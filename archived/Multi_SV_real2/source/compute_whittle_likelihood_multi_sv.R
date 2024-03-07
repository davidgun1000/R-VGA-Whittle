compute_whittle_likelihood_multi_sv <- function(Y, fourier_freqs, periodogram,
                                                params, use_tensorflow = F) {
  
  Tfin <- nrow(Y)
  
  Phi <- params$Phi
  Sigma_eta <- params$Sigma_eta
  # Sigma_eps <- params$Sigma_eps
  
  d <- as.integer(dim(Phi)[1])
  
  freq <- fourier_freqs
  I <- periodogram
  
  # ## Fourier frequencies
  # k <- seq(-ceiling(Tfin/2)+1, floor(Tfin/2), 1)
  # k_in_likelihood <- k [k >= 1 & k <= floor((Tfin-1)/2)]
  # freq <- 2 * pi * k_in_likelihood / Tfin
  # 
  # # ## astsa package
  # Z <- log(Y^2) - rowMeans(log(Y^2))
  # fft_out <- mvspec(t(Z), detrend = F, plot = F)
  # I_all <- fft_out$fxx
  
  # Spectral density matrix
  Phi_0 <- diag(d)
  Phi_1 <- Phi
  Theta <- diag(d)
  
  log_likelihood <- 0
  if (use_tensorflow) { # compute the likelihood for all frequencies at once
    nfreqs <- as.integer(length(freq))
    Phi_0_tf <- tf$eye(d)
    Phi_0_reshaped <- tf$reshape(Phi_0, c(1L, dim(Phi_0)))
    Phi_0_tiled <- tf$tile(Phi_0_reshaped, c(nfreqs, 1L, 1L))
    
    Phi_1_tf <- tf$Variable(Phi_1)
    Phi_1_reshaped <- tf$reshape(Phi_1, c(1L, dim(Phi_1)))
    Phi_1_tiled <- tf$tile(Phi_1_reshaped, c(nfreqs, 1L, 1L))
    
    freq_tf <- tf$Variable(freq)
    I_tf <- tf$Variable(I[,,1:nfreqs])
    I_tf <- tf$transpose(I_tf)
    
    # freq_tf <- tf$Variable(exp(-1i * freq))
    freq_reshape <- tf$reshape(freq_tf, c(dim(freq_tf), 1L, 1L))
    
    exp_1i <- tf$exp(tf$multiply(-1i, tf$cast(freq_tf, "complex128")))
    exp_1i_reshape <- tf$reshape(exp_1i, c(dim(exp_1i), 1L, 1L))
    Phi_mat <- tf$cast(Phi_0_tiled, "complex128") - tf$multiply(tf$cast(Phi_1_tiled, "complex128"), exp_1i_reshape)
    # Phi_inv <- tf$cast(Phi_0_tiled, "complex128") - tf$multiply(tf$cast(Phi_1_tiled, "complex128"), 
    #                                                          tf$exp(tf$multiply(-1i, tf$cast(freq_tf, "complex128"))))
    Phi_inv_tf <- tf$linalg$inv(Phi_mat)
    Phi_inv_H_tf <- tf$math$conj(tf$transpose(Phi_inv_tf, perm = c(0L, 2L, 1L))) # perm is to make sure transposes are done on the 2x2 matrix not on the batch dimension
    
    # Theta_tf <- tf$cast(tf$eye(d))
    # Theta_reshaped <- tf$reshape(Theta_tf, c(dim(Theta_tf), 1L, 1L))
    # Theta_tiled <- tf$tile(Theta_reshaped, c(nfreqs, 1L, 1L))
    Theta_tf <- tf$cast(Phi_0_tiled, "complex128")
    Sigma_eta_tf <- tf$Variable(Sigma_eta, dtype = "complex128")
    
    spec_dens_X_tf <- tf$linalg$matmul(tf$linalg$matmul(tf$linalg$matmul(tf$linalg$matmul(Phi_inv_tf, Theta_tf), Sigma_eta_tf), Theta_tf), Phi_inv_H_tf)
    
    spec_dens_Xi_tf <- tf$multiply(pi^2/2, tf$eye(d))
    spec_dens_Xi_reshape <- tf$reshape(spec_dens_Xi_tf, c(1L, dim(spec_dens_Xi_tf)))
    spec_dens_Xi_tiled <- tf$tile(spec_dens_Xi_reshape, c(nfreqs, 1L, 1L))
    
    spec_dens_tf <- spec_dens_X_tf + tf$cast(spec_dens_Xi_tiled, "complex128")
    
    inv_spec_dens_I <- tf$matmul(tf$linalg$inv(spec_dens_tf), I_tf)
    part2_tf <- tf$reduce_sum(tf$linalg$diag_part(inv_spec_dens_I), 1L)
    
    det_spec_dens_tf <- tf$math$reduce_prod(tf$linalg$eigvals(spec_dens_tf), axis = 1L)
    part1_tf <- tf$math$log(det_spec_dens_tf)
    
    log_likelihood_tf <- -(part1_tf + part2_tf)
    
    if (any(as.vector(tf$math$imag(log_likelihood_tf)) > 1e-10)) {
      print("Warning: Imaginary part of the log likelihood is non-zero")
      browser()
    } else {
      log_likelihood <- sum(as.vector(tf$math$real(log_likelihood_tf)))
    }

    spec_dens_X <- as.matrix(spec_dens_X_tf) # for the return object
    
  } else {
    
    spec_dens_X <- list()
    
    for (k in 1:length(freq)) {
      Phi_inv <- solve(Phi_0 - Phi_1 * exp(- 1i * freq[k]))
      Phi_inv_H <- Conj(t(Phi_inv))
      
      # test <- solve(diag(2) - Phi * exp(- 1i * freq[k])) %*% Sigma_eta %*% 
      #   solve(diag(2) - t(Phi) * exp(1i * freq[k]))
      # M <- Phi_0 - Phi_1 * exp(- 1i * freq[k])
      # M_H <- Conj(t(M))
      # M_H_inv <- solve(M_H)
      
      spec_dens_X[[k]] <- Phi_inv %*% Theta %*% Sigma_eta %*% Theta %*% Phi_inv_H
      # spec_dens_X[[k]] <- Phi_inv %*% Theta %*% Sigma_eta %*% Theta %*% Phi_inv_H
      
      spec_dens_Xi <- diag(pi^2/2, d)
      
      spec_dens <- spec_dens_X[[k]] + spec_dens_Xi  
      
      part2 <- sum(diag(solve(spec_dens) %*% I[, , k]))
      # test <- Conj(t(I[,,k])) %*% solve(spec_dens) %*% I[,,k]
      # part2 <- sum(diag(solve(spec_dens) %*% I_all[[k]]))
      
      # log(det(spec_dens))
      
      if (d == 2) {
        det_spec_dens <- prod(diag(spec_dens)) - spec_dens[1,2] * spec_dens[2,1]
      } else {
        det_spec_dens <- prod(eigen(spec_dens, only.values = T)$values)
      }
      
      part1 <- log(det_spec_dens)
      
      log_likelihood <- log_likelihood - (part1 + part2)
      # test <- spec_dens[1, 2] * spec_dens[2, 1] - spec_dens[1, 1] * spec_dens[2, 2]
    }
    
    if (Im(log_likelihood) > 1e-10) { # if imaginary part is non zero
      print("Warning: Imaginary part of the log likelihood is non-zero")
      browser()
    } else { # if imaginary part is effectively zero, get rid of it
      log_likelihood <- Re(log_likelihood)
    }
  }
    
    
  return(list(log_likelihood = log_likelihood,
              spec_dens_X = spec_dens_X))
  # return(spec_dens_X)
}


# compute_whittle_likelihood_multi_sv <- function(Y, params) {
#   
#   Tfin <- ncol(Y)
#   
#   A <- params$A
#   Sigma_eta <- params$Sigma_eta
#   
#   ## Fourier frequencies
#   k <- seq(-ceiling(Tfin/2)+1, floor(Tfin/2), 1)
#   k_in_likelihood <- k [k >= 1 & k <= floor((Tfin-1)/2)]
#   freq <- 2 * pi * k_in_likelihood / Tfin
#   
#   Y_tilde <- log(Y^2) - rowMeans(log(Y^2))
#   
#   # J <- fft(Y)
#   # J <- t(J)
#   # I_omega <- 1/(2*pi*Tfin) * J %*% Conj(t(J)) # Should this be TxT?
#   
#   # J_test <- 0
#   # for (t in 1:Tfin) {
#   #   J_test <- J_test + Y[, t] * exp(-1i * freq[1] * t) 
#   # }
#   # test <- exp(-1i * freq[1] * (0:(Tfin-1)))
#   # test2 <- t(cbind(test, test))
#   # test2 <- t(cbind(test, test))
#   # J_all <- Y * test2 # but there is no Y_0???
#   # J_test2 <- rowSums(J_all)
#   
#   ##########################################
#   
#   # J_list <- list()
#   # for (k in 1:length(freq)) {
#   #   test <- exp(-1i * freq[k] * (0:(Tfin-1))) #(1:Tfin))
#   #   # test2 <- t(cbind(test, test))
#   #   # J_manual <- X[, 1] * exp(-1i * freq[1] * 1) # need to check the fft calculation manually
#   #   J_all <- Y_tilde * t(cbind(test, test)) # but there is no Y_0???
#   #   J_list[[k]] <- rowSums(J_all)
#   # }
#   # 
#   # # # J_vec <- lapply(J, function(j) j[1])
#   # # # re <- Re(unlist(J_vec))
#   # # fft_out <- fft(Y_tilde)
#   # # J_fft <- fft_out[, 2:501]
#   # # # head(Re(fft_out[1, 2:501]))
#   # # J_list <- lapply(seq_len(ncol(J_fft)), function(i) J_fft[, i])
#   # 
#   # I_all <- lapply(J_list, function(M, Tfin) 1/(2*pi*Tfin) * M %*% Conj(t(M)), Tfin = Tfin)
#   # 
#   
#   # ## astsa package
#   fft_out <- mvspec(t(Y_tilde))
#   I_all <- fft_out$fxx
#   
#   # Spectral density matrix
#   Phi_0 <- diag(2)
#   Phi_1 <- A
#   Theta <- diag(2)
#   
#   log_likelihood <- 0
#   for (k in 1:length(freq)) {
#     Phi_inv <- solve(Phi_0 - Phi_1 * exp(- 1i * freq[k]))
#     Phi_inv_H <- Conj(t(Phi_inv))
#     
#     # M <- Phi_0 - Phi_1 * exp(- 1i * freq[k])
#     # M_H <- Conj(t(M))
#     # M_H_inv <- solve(M_H)
#     
#     # spec_dens_X <- 1/(2*pi) * Phi_inv %*% Theta %*% Sigma_eta %*% Theta %*% Phi_inv_H
#     spec_dens_X <- Phi_inv %*% Theta %*% Sigma_eta %*% Theta %*% Phi_inv_H
#     
#     # spec_dens_Xi <- 1/(2*pi) * diag(pi^2/2, 2)
#     spec_dens_Xi <- diag(pi^2/2, 2)
#     
#     spec_dens <- spec_dens_X + spec_dens_Xi  
#       
#     part2 <- sum(diag(solve(spec_dens) %*% I_all[, , k]))
#     
#     # log(det(spec_dens))
#     
#     det_spec_dens <- prod(eigen(spec_dens)$values)
#     part1 <- log(det_spec_dens)
#     
#     log_likelihood <- log_likelihood - (part1 + part2)
#     # test <- spec_dens[1, 2] * spec_dens[2, 1] - spec_dens[1, 1] * spec_dens[2, 2]
#   }
#   
#   return(log_likelihood)
#    
# }