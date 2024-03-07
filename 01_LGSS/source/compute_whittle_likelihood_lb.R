compute_whittle_likelihood_lb <- function(y, params, I, freq) {
  n <- length(y)
  phi <- params$phi
  sigma_eps <- params$sigma_eps
  sigma_eta <- params$sigma_eta
  
  # ## Calculate the spectral density of x, which is an AR(1) process
  # k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
  # # k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)] # only use frequencies from 1 onwards in the likelihood calculation
  # k_in_likelihood <- k[k >= 1 & k <= i]
  # freq_in_likelihood <- 2 * pi * k_in_likelihood / n
  
  # spectral_dens_x <- 1/(2*pi) * sigma_eta^2 / (1 + phi^2 - 2 * phi * cos(freq_in_likelihood))
  spectral_dens_x <- sigma_eta^2 / (1 + phi^2 - 2 * phi * cos(freq))
  spectral_dens_eps <- sigma_eps^2 #/ (2*pi)
  spectral_dens_y <- spectral_dens_x + spectral_dens_eps
  
  ## Fourier transform of the series
  # fourier_transf <- fft(y)
  # I_omega <- 1/n * Mod(fourier_transf)^2
  
  ## Calculate the Whittle likelihood
  part1 <- log(spectral_dens_y)
  # part2 <- I[1:length(freq) + 1] / spectral_dens_y
  part2 <- I / spectral_dens_y
  
  # part2 <- I_omega / spectral_dens_y
  
  log_whittle <- - sum(part1 + part2)
  
  return(log_whittle)
}