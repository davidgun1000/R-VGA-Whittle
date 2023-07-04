## Plot the likelihood up to the kth frequency
setwd("~/R-VGA-Whittle/LGSS/")
rm(list = ls())


library(mvtnorm)
source("./source/compute_whittle_likelihood_lb.R")

## Flags
date <- "20230525"
regenerate_data <- F
save_data <- F

## R-VGA flags
use_tempering <- T
reorder_freq <- T
decreasing <- T
transform <- "arctanh"

## MCMC flags
adapt_proposal <- T

## True parameters
sigma_eps <- 0.5 # measurement error var
sigma_eta <- 0.7 # process error var
phi <- 0.9

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

# Generate true process x_1:T
n <- 1000
# times <- seq(0, 1, length.out = iters)

if (regenerate_data) {
  print("Generating data...")
  x <- c()
  x[1] <- 1
  set.seed(2023)
  for (t in 2:n) {
    x[t] <- phi * x[t-1] + rnorm(1, 0, sigma_eta)
  }
  
  # Generate observations y_1:T
  y <- x + rnorm(n, 0, sigma_eps)
  
  ## Plot true process and observations
  # par(mfrow = c(1, 1))
  # plot(x, type = "l", main = "True process")
  # points(y, col = "cyan")
  
  lgss_data <- list(x = x, y = y, phi = phi, sigma_eps = sigma_eps, sigma_eta = sigma_eta)
  if (save_data) {
    saveRDS(lgss_data, file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  }
} else {
  print("Reading saved data...")
  lgss_data <- readRDS(file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  y <- lgss_data$y
  x <- lgss_data$x
  phi <- lgss_data$phi
  sigma_eps <- lgss_data$sigma_eps
  sigma_eta <- lgss_data$sigma_eta
}

## Fourier frequencies
k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
freq <- 2 * pi * k_in_likelihood / n

## Fourier transform of the observations
fourier_transf <- fft(y)
periodogram <- 1/n * Mod(fourier_transf)^2
I <- periodogram[k_in_likelihood + 1]


llh <- c()

for (i in 1:length(freq)) {
  llh[i] <- compute_whittle_likelihood_lb(y = y, 
                                          params = list(phi = phi,
                                                        sigma_eta = sigma_eta,
                                                        sigma_eps = sigma_eps), 
                                          I = I[1:i], 
                                          freq = freq[1:i])
}

plot(llh, type = "l")
