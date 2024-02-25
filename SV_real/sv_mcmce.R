## Stochastic volatility model
setwd("~/R-VGA-Whittle/SV/")

library(mvtnorm)

# source("./source/compute_whittle_likelihood_sv.R")
source("./source/run_mcmc_sv.R")
source("./source/kalmanFilter.R")
source("./source/particleFilter.R")

## Flags
rerun_mcmce <- T
save_mcmce_results <- F

## Generate data
mu <- 0
phi <- 0.9
sigma_eta <- 0.7
sigma_eps <- 1

set.seed(2023)
x1 <- rnorm(1, mu, sigma_eta^2 / (1 - phi^2))
n <- 1000
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

## Test KF and PF
# paramsKF <- list(phi = phi, sigma_eta = sigma_eta, sigma_eps = sqrt(pi^2/2))
# 
# kf_out <- kalmanFilter(params = paramsKF, state_prior_mean = 0,
#                    state_prior_var = 1,
#                    observations = log(y^2), iters = length(y))
# N <- 500
# paramsPF <- list(phi = phi, sigma_eta = sigma_eta, sigma_xi = sqrt(pi^2/2))
# pf_out <- particleFilter(y = y, N = N, iniState = 0, param = paramsPF)
# 
# par(mfrow = c(1,1))
# plot_range <- 1:200
# plot(x[plot_range], type = "l")
# # lines(kf_out$kf_mean[plot_range], col = "red")
# lines(pf_out$state_mean[plot_range], col = "blue")
# 
# browser()
# 

# Likelihood surface
# phi_grid <- seq(0.1, 0.99, length.out = 200)
# llh_kf <- llh_pf <- c()
# N <- 100
# for (k in 1:length(phi_grid)) {
#   paramsKF <- list(phi = phi_grid[k], sigma_eta = sigma_eta, sigma_eps = sqrt(pi^2/2))
# 
#   kf <- kalmanFilter(params = paramsKF, state_prior_mean = 0,
#                           state_prior_var = 1,
#                           observations = log(y^2), iters = length(y))
#   llh_kf[k] <- kf$log_likelihood
# 
#   paramsPF <- list(phi = phi_grid[k], sigma_eta = sigma_eta, sigma_xi = sqrt(pi^2/2))
#   pf_out <- particleFilter(y = y, N = N, iniState = 0, param = paramsPF)
#   llh_pf[k] <- pf_out$log_likelihood
# 
# }
# 
# par(mfrow = c(2,1))
# plot(phi_grid, llh_kf, type = "l")
# abline(v = phi, lty = 2)
# abline(v = phi_grid[which.max(llh_kf)], lty = 2, col = "red")
# plot(phi_grid, llh_pf, type = "l")
# abline(v = phi, lty = 2)
# abline(v = phi_grid[which.max(llh_pf)], lty = 2, col = "red")
# browser()

# ## MCMC with likelihood estimated via KF/PF
# 
# n_post_samples <- 10000
# burn_in <- 5000
# MCMC_iters <- n_post_samples + burn_in
# 
# prior_mean <- rep(0, 3)
# prior_var <- diag(1, 3)
# 
# if (rerun_mcmce) {
#   mcmce_results <- run_mcmc_sv(y, #sigma_eta, sigma_eps, 
#                                iters = MCMC_iters, burn_in = burn_in,
#                                prior_mean = prior_mean, prior_var = prior_var,  
#                                state_ini_mean = state_ini_mean, state_ini_var = state_ini_var,
#                                adapt_proposal = T, use_whittle_likelihood = F)
#   
#   if (save_mcmce_results) {
#     saveRDS(mcmce_results, mcmcw_filepath)
#   }
# } else {
#   mcmce_results <- readRDS(mcmce_filepath)
# }
# 
# mcmce.post_samples_phi <- as.mcmc(mcmce_results$post_samples$phi[-(1:burn_in)])
# mcmce.post_samples_eta <- as.mcmc(mcmce_results$post_samples$sigma_eta[-(1:burn_in)])
# mcmce.post_samples_xi <- as.mcmc(mcmce_results$post_samples$sigma_xi[-(1:burn_in)])
# 
# par(mfrow = c(1,3))
# plot(density(mcmce.post_samples_phi), main = "Posterior of phi", 
#      col = "blue", lty = 2, lwd = 1.5)
# 
# par(mfrow = c(1,3))
# plot(density(mcmce.post_samples_eta), main = "Posterior of sigma_eta", 
#      col = "blue", lty = 2, lwd = 1.5)
# 
# par(mfrow = c(1,3))
# plot(density(mcmce.post_samples_xi), main = "Posterior of sigma_xi", 
#      col = "blue", lty = 2, lwd = 1.5)

################################################################################
# ###
# ## MCMC with likelihood estimated via particle filter
# ###
# 
# 
# particles <- matrix(NA, nrow = N, ncol = n + 1)
# weights <- normalisedW <- matrix(NA, nrow = N, ncol = n + 1)
# indices <- matrix(NA, nrow = N, ncol = n + 1)
# logLikelihood <- 0
# covmats <- list()
# 
# # Initialise
# weights[, 1] <- normalisedW[, 1] <- 1/N
# indices[, 1] <- 1:N
# 
# filteredState <- c()
# filteredState[1] <- iniState
# particles[, 1] <- iniState
# 
# # phi <- param$phi
# # sigma_v <- param$sigma_v
# # sigma_e <- param$sigma_e
# 
# for (t in 2:n) {
#   # (i) Resample
#   newIndices <- sample(N, replace = TRUE, prob = normalisedW[, t-1])
#   
#   # particles[, t] <- particles[newIndices, t-1]
#   # filteredState[, t] <- filteredState[newIndices, t-1]
#   indices[, t] <- newIndices
#   
#   # (ii) Propagate
#   particles[, t] <- phi * particles[newIndices, t-1] + rnorm(N, 0, params$sigma_eta) 
#   
#   # (iii) Compute weight
#   pseudo_y <- particles[, t] + rnorm(N, 0, params$sigma_xi)
#   
#   weights[, t] <- dnorm(y_tilde[t+1], mean = pseudo_y, sd = rep(params$sigma_xi, N),
#                         log = TRUE)
#   
#   # (iv) Normalise weight
#   maxWeight <- max(weights[, t])
#   expWeights <- exp(weights[, t] - maxWeight) #exp(weights - maxWeights)
#   normalisedW[, t] <- expWeights / sum(expWeights) 
#   
#   # (v) Compute likelihood
#   predictiveLike <- maxWeight + log(sum(expWeights)) - log(N)
#   logLikelihood <- logLikelihood + predictiveLike
#   
#   # (vi) Estimate state
#   filteredState[t] <- mean(particles[, t])
#   covmats[[t]] <- 1/(N - 1) * tcrossprod(particles[, t] - mean(particles[, t]))
#   
# }
# 
# 

## Implement particle filter here
N <- 500
y_tilde <- log(y^2)
params <- list(phi = phi, sigma_eta = sigma_eta, sigma_xi = sqrt(pi^2/2))

weights <- list()
X <- list()
x_estimate <- c()
log_likelihood <- c()
indices <- list()
indices[[1]] <- 1:N

iniState <- 0
# t = 1:
## Sample particles
X[[1]] <- rep(0, N) #rnorm(N, 0, sqrt(params$sigma_eta^2 / (1 - params$phi^2)))

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
plot_range <- 1:200
# x_mean <- sapply(X, mean)
plot(x[plot_range], type = "l")
# lines(filteredState[plot_range], col = "red")
lines(x_estimate[plot_range], col = "red")


## Someone else's code
obs <- log(y^2)
Tfin <- length(obs)
N <- 500

# create x and weight matrices
x_pf <- matrix(nrow =  N, ncol = Tfin)
weights <- normalisedW <- matrix(nrow =  N, ncol = Tfin)
# intial (at t=1):
# draw X from prior distribution
sx <- sigma_eta
sy <- sqrt(pi^2/2)
x_pf[, 1] <- rnorm(N, 0, sx)
# calculate weights, i.e. probability of evidence given sample from X
weights[, 1] <- dnorm(obs[1], x_pf[, 1], sy)

# normalise weights 
weights[, 1] <- weights[, 1]/sum(weights[, 1])

# (v) Compute likelihood
predictiveLike <- maxWeight + log(sum(expWeights)) - log(N)
logLikelihood <- logLikelihood + predictiveLike

# weighted resampling with replacement. This ensures that X will converge to the true distribution
x_pf[, 1] <- sample(x_pf[, 1], replace = TRUE, size = N, prob = weights[, 1]) 

for (t in seq(2, Tfin)) {
  # predict x_{t} from previous time step x_{t-1}
  # based on process (transition) model
  x_pf[, t] <- rnorm(N, phi * x_pf[, t-1], sx)
  # calculate  and normalise weights
  # weights[, t] <- dnorm(obs[t], x_pf[, t], sy)
  pseudo_xi <- obs[t] - x_pf[, t]
  weights[, t] <- dchisq(exp(pseudo_xi), 1) * exp(pseudo_xi)
  
  weights[, t] <- weights[, t]/sum(weights[, t])
  # weighted resampling with replacement
  x_pf[, t] <- sample(x_pf[, t], replace = TRUE, size = N, prob = weights[, t]) 
}

x_means <- apply(x_pf, 2, mean)
x_quantiles <- apply(x_pf, 2, function(x) quantile(x, probs = c(0.025, 0.975)))

par(mfrow = c(1,1))
plot(x, type = "l")
lines(x_means, col = "red") # someone else's code
lines(x_estimate, col = "blue") # my new code
lines(pf_out$filteredState, col = "green") # my old code from last year
