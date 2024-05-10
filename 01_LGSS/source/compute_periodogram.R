compute_periodogram <- function(obs) {
    ## Fourier frequencies
    k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
    k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
    freq <- 2 * pi * k_in_likelihood / n

    ## Fourier transform of the observations
    fourier_transf <- fft(y)
    periodogram <- 1/n * Mod(fourier_transf)^2
    I <- periodogram[k_in_likelihood + 1]

# total_power <- sum(I)
# cumsum_power <- cumsum(I)
# ind <- which(cumsum_power >= 0.8*total_power)[1]

# browser()
# half_power <- max(10*log10(I))/2
# ind <- which(10*log10(I) >= half_power)[1]

# png("plots/periodogram.png")
# plot(10*log10(I), type = "l", main = "Periodogram on log10 scale",
#         xlab = "Frequency", ylab = "10*log10(I) (dB)")
# abline(v = 1000, col = "red", lty = 2, lwd = 2)
# # abline(v = ind, col = "blue", lty = 2, lwd = 2)
# abline(h = -3, col = "red", lty = 2, lwd = 2)
# dev.off()
# browser()

    return(list(freq = freq, periodogram = I))
}


