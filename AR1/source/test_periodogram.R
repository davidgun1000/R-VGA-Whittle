## Test on periodogram calculation
setwd("~/R-VGA-Whittle/AR1/")
## Model parameters 
phi <- 0.9
sigma_e <- 0.1
n <- 1000 # time series length

## Generate AR(1) series
x0 <- 0 #1
x <- c()
x[1] <- x0
set.seed(2023)
for (t in 2:n) {
  x[t] <- phi * x[t-1] + rnorm(1, 0, sigma_e) 
}
rvgaw_data <- list(x = x, phi = phi, sigma_e = sigma_e)


## Manual 
k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
freq <- 2 * pi * k_in_likelihood / n

t <- 1:n
J <- c()
for (j in 1:length(freq)) {
  J[j] <- sum(x * exp(-(1i) * freq[j] * t)) # is the 1/sqrt(2*pi) unnecessary??
}
I <- 1/n * Mod(J)^2

## Using fft() in R
fourier_transf <- fft(x)
periodogram <- 1/n * Mod(fourier_transf)^2
I2 <- periodogram[k_in_likelihood + 1] # shift the indices because the periodogram is calculated from k = 0, whereas in Whittle likelihood we start from k = 1

head(I)
head(I2)
