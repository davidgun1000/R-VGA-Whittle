rm(list = ls())
setwd("~/R-VGA-Whittle/Multi_SV/")

library(beyondWhittle)
library(mvtnorm)

source("./source/run_rvgaw_multi_sv.R")
source("./source/run_mcmc_multi_sv.R")
source("./source/compute_whittle_likelihood_multi_sv.R")
# source("./source/run_mcmc_sv.R")
# source("./source/compute_whittle_likelihood_sv.R")
source("./source/construct_prior.R")
source("./source/map_functions.R")
# source("./archived/compute_partial_whittle_likelihood.R")
# source("./source/compute_grad_hessian.R")

result_directory <- "./results/"

## Flags
date <- "20230814"
regenerate_data <- T
save_data <- F
use_cholesky <- F # use lower Cholesky factor to parameterise Sigma_eta

#######################
##   Generate data   ##
#######################

dataset <- "3"

Tfin <- 1000
d <- 3

if (regenerate_data) {
  
  sigma_eta1 <- 0.9#4
  sigma_eta2 <- 1.5 #1.5
  sigma_eta3 <- 1.2 #1.5
  
  Sigma_eta <- diag(c(sigma_eta1, sigma_eta2, sigma_eta3))
  Sigma_eps <- diag(d)
  
  if (dataset == "0") {
    # Phi <- diag(c(0.7, 0.8, 0.9))
    Phi <- diag(c(0.9, 0.2, -0.8))
  } else if (dataset == "3") {
    Phi <- matrix(c(0.9, 0.0, -0.1,
                    -0.6, 0.2, 0.5,
                    1.0, 0.3, -0.8), 3, 3, byrow = T)
    # diag(Phi) <- c(0.7, 0.8, 0.9)
  } else if (dataset == "2") {
    Phi <- matrix(c(-0.3, -0.1,  0.4,
                    -0.2,  0.9,  0.8,
                    -0.2, -0.7, -0.8), 3, 3, byrow = T)
  } else { # generate a random Phi matrix using the mapping from unconstrained to constrained VAR(1) coef 
    A <- matrix(rnorm(d^2), d, d)
    Phi <- backward_map(A, Sigma_eta)
    Phi <- round(Phi, digits = 1)
  }
  
  set.seed(2022)
  x1 <- rmvnorm(1, rep(0, d), compute_autocov_VAR1(Phi, Sigma_eta))
  X <- matrix(NA, nrow = Tfin+1, ncol = length(x1)) # x_0:T
  X[1, ] <- x1
  Y <- matrix(NA, nrow = Tfin, ncol = length(x1)) # y_1:T
  # Y[, 1] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
  for (t in 1:Tfin) {
    X[t+1, ] <- Phi %*% X[t, ] + t(rmvnorm(1, rep(0, d), Sigma_eta))
    V <- diag(exp(X[t+1, ]/2))
    Y[t, ] <- V %*% t(rmvnorm(1, rep(0, d), Sigma_eps))
  }
  
  multi_sv_data <- list(X = X, Y = Y, Phi = Phi, Sigma_eta = Sigma_eta, 
                        Sigma_eps = Sigma_eps)
  
  if (save_data) {
    saveRDS(multi_sv_data, file = paste0("./data/trivariate_sv_data_Tfin", Tfin, "_", date, "_", dataset, ".rds"))
  }
} else {
  multi_sv_data <- readRDS(file = paste0("./data/trivariate_sv_data_Tfin", Tfin, "_", date, "_", dataset, ".rds"))
  
  X <- multi_sv_data$X
  Y <- multi_sv_data$Y
  Phi <- multi_sv_data$Phi
  Sigma_eta <- multi_sv_data$Sigma_eta
  Sigma_eps <- multi_sv_data$Sigma_eps
}

par(mfrow = c(d,1))
plot(X[, 1], type = "l")
plot(X[, 2], type = "l")
plot(X[, 3], type = "l")

plot(Y[, 1], type = "l")
plot(Y[, 2], type = "l")
plot(Y[, 3], type = "l")

## Construct initial distribution/prior
prior <- construct_prior(data = t(Y))
prior_mean <- prior$prior_mean
prior_var <- prior$prior_var

param_dim <- length(prior_mean)

# ## Demean data
# Y <- 
  
## Run gibbs sampling
mcmc <- gibbs_var(data=Y, ar.order=1, Ntotal=10000, burnin=4000, thin=1) #,
                  # beta.mu = prior_mean[1:(d^2)], beta.Sigma = prior_var[1:(d^2), 1:(d^2)])

# Plot spectral estimate, credible regions and periodogram on log-scale
plot(mcmc, log=T)

par(mfrow = c(d,d))
indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
for (k in 1:nrow(indices)) {
  i <- indices[k, 1]
  j <- indices[k, 2]
  
  plot(density(mcmc$beta[k, ]))
  abline(v = Phi[i, j], lty = 2)
}
