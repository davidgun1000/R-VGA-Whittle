## Stochastic volatility model
setwd("~/R-VGA-Whittle/SV/")

library(mvtnorm)
library(coda)

source("compute_whittle_likelihood_sv.R")

## Generate data
mu <- 0
sigma_eta <- 0.5
phi <- 0.9
x1 <- rnorm(1, mu, sigma_eta^2 / (1 - phi^2))
n <- 1000
x <- c()
x[1] <- x1

for (t in 2:n) {
  x[t] <- mu + phi * (x[t-1] - mu) + sigma_eta * rnorm(1, 0, 1)
}

eps <- rnorm(n, 0, 1)
y <- exp(x/2) * eps

par(mfrow = c(2,1))
plot(y, type = "l")
plot(x, type = "l")

###
## MCMC with likelihood estimated via particle filter
###

y_tilde <- log(y^2)

weights <- list()

## Implement particle filter here
N <- 100
params <- list(phi = phi, sigma_eta = sigma_eta, sigma_xi = pi^2/2)
X <- list()
x_estimate <- c()
log_likelihood <- c()
indices <- list()
indices[[1]] <- 1:N

# t = 1:
## Sample particles
X[[1]] <- rnorm(N, 0, sqrt(params$sigma_eta^2 / (1 - params$phi^2)))

## Compute weights
weights[[1]] <- dnorm(rep(y_tilde[1], N), X[[1]], rep(params$sigma_xi, N))

## Normalise weights
norm_weights <- weights[[1]]/sum(weights[[1]])

## Estimate likelihood
log_likelihood[1] <- log(mean(weights[[1]]))

# Repeat for t > 1
for (t in 2:n) {
  ## Resample
  resampled_indices <- sample(N, replace = T, prob = norm_weights)
  indices[[t]] <- resampled_indices
  X_resampled <- X[[t-1]][resampled_indices]
  
  ## Propagate
  X[[t]] <- phi * X_resampled + rnorm(N, 0, params$sigma_eta)
  
  ## Compute weights
  weights[[t]] <- dnorm(rep(y_tilde[t], N), X[[t]], rep(params$sigma_xi, N))
  
  ## Normalise weights
  norm_weights <- weights[[t]]/sum(weights[[t]])
  
  ## Estimate likelihood
  log_likelihood[t] <- log(mean(weights[[t]]))
  
  ## Estimate state
  x_estimate[t] <- mean(X[[t]])
  
}

par(mfrow = c(1,1))
# x_mean <- sapply(X, mean)
plot(x_estimate, type = "l")
lines(x, col = "red")
