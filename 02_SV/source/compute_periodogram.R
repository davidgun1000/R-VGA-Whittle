compute_periodogram <- function(obs) {
  
  y <- obs
  n <- length(y)
  ## Fourier frequencies
  k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
  k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
  freq <- 2 * pi * k_in_likelihood / n
  
  ## Fourier transform of the observations
  #y_tilde <- log(y^2) - mean(log(y^2))
  
  fourier_transf <- fft(y)
  periodogram <- 1/n * Mod(fourier_transf)^2
  I <- periodogram[k_in_likelihood + 1]
  
# plot(10*log10(I[1:1000]), type = "l", main = "Periodogram on log10 scale",
#       xlab = "Frequency", ylab = "10*log10(I) (dB)")
# abline(v = 100, col = "red", lty = 2, lwd = 2)
# browser()
# which(I >= half_power)

# half_power <- 10*log10(max(I)/2)
# ind <- which(10*log10(I) >= half_power)[1]

# total_power <- sum(I)
# cumsum_power <- cumsum(I)
# ind <- which(cumsum_power >= 0.5*total_power)[1]

# png("plots/periodogram.png")
# plot(10*log10(I), type = "l", main = "Periodogram on log10 scale",
#         xlab = "Frequency", ylab = "10*log10(I) (dB)")
# abline(v = 100, col = "red", lty = 2, lwd = 2)
# abline(v = ind, col = "blue", lty = 2, lwd = 2)
# # abline(h = -3, col = "red", lty = 2, lwd = 2)
# dev.off()
# browser()
  return(list(freq = freq, periodogram = I))
}