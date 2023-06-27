calculate_whittle_likelihood <- function(series, phi, sigma_e = NULL) {
  n <- length(series)
  
  ## Calculate the spectral density of an AR(1)
  k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
  k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
  freq_in_likelihood <- 2 * pi * k_in_likelihood / n
  # spectral_dens <- 1/(2*pi) * sigma_e^2 / (1 + phi^2 - 2 * phi * cos(freq_in_likelihood))
  spectral_dens <- sigma_e^2 / (1 + phi^2 - 2 * phi * cos(freq_in_likelihood))
  
  ## Fourier transform of the series
  fourier_transf <- fft(x)
  I_omega <- 1/n * Mod(fourier_transf)^2
  # I_omega <- 1/(2*pi*n) * Mod(fourier_transf)^2
  
  
  ## Calculate the Whittle likelihood
  part1 <- log(spectral_dens)
  # part2 <- I_omega[k_in_likelihood + 1] /(n/2 * spectral_dens)
  part2 <- I_omega[k_in_likelihood + 1] /(spectral_dens)
  
  # part2 <- I_omega/spectral_dens
  
  log_whittle <- - sum(part1 + part2)
  
  return(log_whittle)
}

## Test by comparing with the usual likelihood (use KF or EnKF here)
calculate_ar1_likelihood2 <- function(series, phi, sigma_e) {
  log_likelihood <- c()
  log_likelihood[1] <- dnorm(x[1], 0, sqrt(sigma_e^2 / (1 - phi^2)), log = T) # assume x1 ~ U[-10, 10]
  for (t in 2:n) {
    log_likelihood <- log_likelihood + dnorm(series[t], phi * series[t-1], sigma_e, log = T)
  }
  return(log_likelihood)
}

calculate_ar1_likelihood <- function(series, phi, sigma_e) {
  n <- length(series)
  d <- c(sqrt(1 - phi^2), rep(1, n-1))
  lower_d <- rep(-phi, n-1)
  rows <- c(1:n, 2:n)
  cols <- c(1:n, 1:(n-1))
  
  L <- sparseMatrix(i = rows, j = cols, x = c(d, lower_d))
  x_tilde <- L %*% (series - 0) # x_tilde = L(x - mu)
  # log_likelihood <- -n/2 * log(2*pi) -n/2 * log(sigma_e^2) + 1/2 * log(det(L)^2) -
  #   1/(2*sigma_e^2) * crossprod(x_tilde)
  detL <- prod(diag(L))
  log_likelihood <- -n/2 * log(2*pi) -n/2 * log(sigma_e^2) + 1/2 * log(detL^2) -
    1/(2*sigma_e^2) * crossprod(x_tilde)
    
  return(as.numeric(log_likelihood))
}