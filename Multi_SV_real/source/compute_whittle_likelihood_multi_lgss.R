compute_whittle_likelihood_multi_lgss <- function(Y, params) {
  
  Tfin <- ncol(Y)
  
  Phi <- params$Phi
  Sigma_eta <- params$Sigma_eta
  Sigma_eps <- params$Sigma_eps
  
  ## Fourier frequencies
  k <- seq(-ceiling(Tfin/2)+1, floor(Tfin/2), 1)
  k_in_likelihood <- k [k >= 1 & k <= floor((Tfin-1)/2)]
  freq <- 2 * pi * k_in_likelihood / Tfin
  
  # ## astsa package
  fft_out <- mvspec(t(Y), plot = F)
  I_all <- fft_out$fxx
  
  # Spectral density matrix
  Phi_0 <- diag(2)
  Phi_1 <- Phi
  Theta <- diag(2)
  
  log_likelihood <- 0
  for (k in 1:length(freq)) {
    Phi_inv <- solve(Phi_0 - Phi_1 * exp(- 1i * freq[k]))
    Phi_inv_H <- Conj(t(Phi_inv))
    
    # M <- Phi_0 - Phi_1 * exp(- 1i * freq[k])
    # M_H <- Conj(t(M))
    # M_H_inv <- solve(M_H)
    
    # spec_dens_X <- 1/(2*pi) * Phi_inv %*% Theta %*% Sigma_eta %*% Theta %*% Phi_inv_H
    spec_dens_X <- Phi_inv %*% Theta %*% Sigma_eta %*% Theta %*% Phi_inv_H
    
    # spec_dens_Xi <- 1/(2*pi) * diag(pi^2/2, 2)
    spec_dens_Xi <- Sigma_eps #diag(pi^2/2, 2)
    
    spec_dens <- spec_dens_X + spec_dens_Xi  
    
    part2 <- sum(diag(solve(spec_dens) %*% I_all[, , k]))
    
    # log(det(spec_dens))
    
    det_spec_dens <- prod(eigen(spec_dens)$values)
    part1 <- log(det_spec_dens)
    
    log_likelihood <- log_likelihood - (part1 + part2)
    # test <- spec_dens[1, 2] * spec_dens[2, 1] - spec_dens[1, 1] * spec_dens[2, 2]
  }
  
  return(log_likelihood)
  
}