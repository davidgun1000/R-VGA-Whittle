## Stochastic volatility model
setwd("~/R-VGA-Whittle/SV/")

# library(mvtnorm)
library(pomp)

# source("./source/compute_whittle_likelihood_sv.R")
# source("./source/run_mcmc_sv.R")
# source("./source/kalmanFilter.R")
# source("./source/particleFilter.R")

## Flags
rerun_mcmce <- T
save_mcmce_results <- F

## Generate data
mu <- 0
phi <- 0.5
sigma_eta <- 1.05
sigma_eps <- 0.7

x1 <- rnorm(1, mu, sigma_eta^2 / (1 - phi^2))
n <- 200
x <- c()
x[1] <- x1

for (t in 2:n) {
  x[t] <- mu + phi * (x[t-1] - mu) + sigma_eta * rnorm(1, 0, 1)
}

eps <- rnorm(n, 0, sigma_eps)
y <- exp(x/2) * eps

par(mfrow = c(1,2))

plot(x, type = "l")
plot(y, type = "l")

process_model <- function(x_prev, phi, sigma_eta) {
  x <- phi * x_prev + rnorm(1, 0, sigma_eta)
  return(x)
}

obs_model <- function(x, sigma_xi) {
  y <- x + rnorm(1, 0, sigma_xi)
  return(x)
}


## PF implemented via pomp
y_df <- data.frame(y = y)
