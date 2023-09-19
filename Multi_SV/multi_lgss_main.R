## Bivariate SV model

setwd("~/R-VGA-Whittle/Multi_SV/")

source("./source/compute_whittle_likelihood_multi_lgss.R")
source("./source/map_functions.R")

library("mvtnorm")
library("astsa")
# library("expm")
library("stcos")

# phi11 <- 0.9
# phi12 <- 0.1
# phi21 <- 0.2
# phi22 <- 0.7

phi11 <- 0.5 #0.9
phi12 <- 0.3 #0.1 # dataset1: 0.1, dataset2 : -0.5
phi21 <- 0.4 #0.2 # dataset1: 0.2, dataset2 : -0.1
phi22 <- 0.6 #0.7

Phi <- matrix(c(phi11, phi12, phi21, phi22), 2, 2, byrow = T)

sigma_eta1 <- 0.6
sigma_eta2 <- 0.3
Sigma_eta <- diag(c(sigma_eta1, sigma_eta2))

sigma_eps1 <- 1
sigma_eps2 <- 1

Sigma_eps <- diag(c(sigma_eps1, sigma_eps2))

Tfin <- 10000
x1 <- c(0, 0)
X <- matrix(NA, nrow = length(x1), ncol = Tfin+1) # x_0:T
X[, 1] <- x1
Y <- matrix(NA, nrow = length(x1), ncol = Tfin) # y_1:T
# Y[, 1] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
for (t in 1:Tfin) {
  X[, t+1] <- Phi %*% X[, t] + t(rmvnorm(1, c(0, 0), Sigma_eta))
  Y[, t] <- X[, t+1] + t(rmvnorm(1, c(0, 0), Sigma_eps))
  # V <- diag(exp(X[, t+1])/2)
  # Y[, t] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
}

# par(mfrow = c(2,1))
# plot(X[1, ], type = "l")
# plot(X[2, ], type = "l")

# plot(Y[1, ], type = "l")
# plot(Y[2, ], type = "l")

## Test: Plot Whittle likelihood on a grid

phi_grid <- seq(0.1, 1, length.out = 50)
sigma_eta_grid <- seq(0.1, 1, length.out = 50)

# test <- compute_whittle_likelihood_multi_sv(Y = Y, 
#                                                params = list(A = A, Sigma_eta = Sigma_eta))
# browser()

llh <- c()

options(warn=2)
for (j in 1:length(phi_grid)) {
  cat("j = ", j, "\n")
  # Phimat <- matrix(c(phi_grid[j], phi12, phi21, phi22), 2, 2, byrow = T)
  Sigma_eta_j <- diag(c(sigma_eta1, sigma_eta_grid[j]))
  llh[j] <- compute_whittle_likelihood_multi_lgss(Y = Y, 
                                            params = list(Phi = Phi, 
                                                          Sigma_eta = Sigma_eta_j, #_j,
                                                          Sigma_eps = Sigma_eps))
}

if (any(Im(llh) - 0 > 1e-10)) { # if imaginary part is effectively zero, get rid of it
  print("Warning: Imaginary part of the log likelihood is non-zero")
} else {
  llh <- Re(llh) 
}

# par(mfrow = c(1,2))
# plot(phi_grid, llh, type = "l")
# abline(v = phi11, col = "black", lty = 2)
# abline(v = phi_grid[which.max(llh)], col = "red", lty = 2)
# legend("topleft", legend = c("True parameter", "arg max (llh)"), 
#        col = c("black", "red"), lty = 2)

par(mfrow = c(1,1))
plot(sigma_eta_grid, llh, type = "l")
abline(v = Sigma_eta[2,2], col = "black", lty = 2)
abline(v = sigma_eta_grid[which.max(llh)], col = "red", lty = 2)
legend("bottomright", legend = c("True parameter", "arg max (llh)"), 
       col = c("black", "red"), lty = 2)

browser()

##########################################
##         R-VGAL implementation        ##
##########################################

## Assume Sigma_eps is known; try to infer Phi and Sigma_eta

## 1. Construct the initial variational distribution/prior -- Minnesota prior
m <- nrow(Y) # dimension of VAR_m(p)
# L_elements <- rnorm(m*(m-1)/2, 0, sqrt(0.1))

param_dim <- m^2 + (m*(m-1)/2 + m)

# Prior mean
prior_mean <- rep(0, param_dim)

# Prior var
ar_out1 <- arima(Y[1, ], order = c(1, 0, 0))
ar_out2 <- arima(Y[2, ], order = c(1, 0, 0)) # use for Minnesota prior
sigma2_estimates <- c(ar_out1$sigma2, ar_out2$sigma2)

indices <- data.frame(i = rep(1:m, each = m), j = rep(1:m, m))

diag_var_A <- c()
  
l = 1
lambda_0 = 1
theta_0 = 0.2
for (k in 1:nrow(indices)) {
  i <- indices[k, 1]
  j <- indices[k, 2]
  
  if (i == j) {
    diag_var_A[k] <- 1
  } else {
    diag_var_A[k] <- (lambda_0 * theta_0 / l)^2 * 
      (sigma2_estimates[i] / sigma2_estimates[j])    
  }
}

## now put the prior of Phi and L together so that
## we have a vector of (Phi, L) parameters
diag_var_L <- rep(0.1, 3)

prior_var <- diag(c(diag_var_A, diag_var_L))

## the plan is basically to generate parameters of A and elements of L from the prior
## then assemble them into the A matrix and the Sigma_eta matrix
## then transform A back into Phi (the original AR parameterisation)
## and use Phi and Sigma_eta to calculate the Whittle likelihood

## 2. Generate parameters from prior
samples <- rmvnorm(1, prior_mean, prior_var) 

### the first 4 elements will be used to construct A
A_sample <- matrix(samples[1:4], 2, 2, byrow = T)

### the last 3 will be used to construct L
L <- diag(exp(samples[5:6]))
L[2,1] <- samples[7]
Sigma_eta_sample <- L %*% t(L)

## 3. Map (A, Sigma_eta) to (Phi, Sigma_eta) using the mapping in Ansley and Kohn (1986)
Phi_sample <- backward_map(A_sample, Sigma_eta_sample)

## 4. Calculate the likelihood
llh <- compute_whittle_likelihood_multi_lgss(Y = Y, 
                                            params = list(Phi = Phi_sample, 
                                                          Sigma_eta = Sigma_eta_sample, #_j,
                                                          Sigma_eps = Sigma_eps))

## 5. Calculate the gradient and Hessian for the likelihood

## Note to self: might be a better idea to get MCMC Whittle running for this, then tackle R-VGAW?