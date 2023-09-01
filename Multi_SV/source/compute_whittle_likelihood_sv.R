compute_whittle_likelihood_sv <- function(y, params) {
  n <- length(y)
  phi <- params$phi
  sigma_eta <- params$sigma_eta
  sigma_xi <- sqrt(pi^2/2) #params$sigma_xi
  
  ## Compute Whittle likelihood
  y_tilde <- log(y^2) - mean(log(y^2))
  
  ## Calculate the spectral density of x, which is an AR(1) process
  k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
  k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)] # only use frequencies from 1 onwards in the likelihood calculation
  freq_in_likelihood <- 2 * pi * k_in_likelihood / n
  # spectral_dens_x <- 1/(2*pi) * sigma_eta^2 / (1 + phi^2 - 2 * phi * cos(freq_in_likelihood))
  spectral_dens_x <- sigma_eta^2 / (1 + phi^2 - 2 * phi * cos(freq_in_likelihood))
  
  spectral_dens_xi <- sigma_xi^2 #/ (2*pi)
  spectral_dens_y_tilde <- spectral_dens_x + spectral_dens_xi
  
  ## Fourier transform of the series
  fourier_transf <- fft(y_tilde)
  I_omega <- 1/n * Mod(fourier_transf)^2
  
  ## Calculate the Whittle likelihood
  part1 <- log(spectral_dens_y_tilde)
  part2 <- I_omega[k_in_likelihood + 1] / spectral_dens_y_tilde
  # part2 <- I_omega / spectral_dens_y_tilde
  
  log_whittle <- - sum(part1 + part2)
  
  return(list(log_likelihood = log_whittle, 
              spec_dens_x = spectral_dens_x, 
              periodogram = I_omega[k_in_likelihood + 1] ))
  # return(spectral_dens_x)
}