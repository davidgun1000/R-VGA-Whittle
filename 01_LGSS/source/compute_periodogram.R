compute_periodogram <- function(obs) {
    ## Fourier frequencies
    k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
    k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
    freq <- 2 * pi * k_in_likelihood / n

    ## Fourier transform of the observations
    fourier_transf <- fft(y)
    periodogram <- 1/n * Mod(fourier_transf)^2
    I <- periodogram[k_in_likelihood + 1]

    return(list(freq = freq, periodogram = I))
}


