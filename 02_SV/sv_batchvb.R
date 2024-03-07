## Batch VB for SV model

setwd("~/R-VGA-Whittle/SV")

library(Deriv)
library(mvtnorm)
source("./source/compute_whittle_likelihood_sv.R")
# source("./source/run_rvgaw_sv_tf.R")

## R-VGA flags
date <- "20230918"
regenerate_data <- T
save_data <- F
use_adam <- F

## Generate data
mu <- 0
phi <- 0.9
sigma_eta <- 0.1
sigma_eps <- 1
kappa <- 2
set.seed(2023)
x1 <- rnorm(1, mu, sigma_eta^2 / (1 - phi^2))
n <- 1000

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

## Generate data
if (regenerate_data) {
  x <- c()
  x[1] <- x1
  
  for (t in 2:(n+1)) {
    x[t] <- phi * x[t-1] + sigma_eta * rnorm(1, 0, 1)
  }
  
  eps <- rnorm(n, 0, sigma_eps)
  y <- kappa * exp(x[2:(n+1)]/2) * eps
  
  par(mfrow = c(1,1))
  plot(x, type = "l")
  
  sv_data <- list(x = x, y = y, phi = phi, sigma_eta = sigma_eta, 
                  sigma_eps = sigma_eps, kappa = kappa)
  
  if (save_data) {
    saveRDS(sv_data, file = paste0("./data/sv_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  }
  
} else {
  print("Reading saved data...")
  sv_data <- readRDS(file = paste0("./data/sv_data_n", n, "_phi", phi_string, "_", date, ".rds"))
}

y <- sv_data$y
x <- sv_data$x
phi <- sv_data$phi
sigma_eta <- sv_data$sigma_eta
sigma_eps <- sv_data$sigma_eps

## Prior
# if (prior_type == "prior1") {
  prior_mean <- c(0, -1) #rep(0,2)
  prior_var <- diag(c(1, 0.5)) #diag(1, 2)
# } else {
#   prior_mean <- c(0, -0) #rep(0,2)
#   prior_var <- diag(c(1, 0.5)) #diag(1, 2)
# }

## Aux functions
compute_periodogram <- function(y) {

  n <- length(y)
  ## Fourier frequencies
  k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
  k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
  freq <- k_in_likelihood / n # 2 * pi * k_in_likelihood / n

  ## Fourier transform of the observations
  y_tilde <- log(y^2) - mean(log(y^2))

  fourier_transf <- fft(y_tilde)
  periodogram <- 1/n * Mod(fourier_transf)^2
  I <- periodogram[k_in_likelihood + 1]

  return(list(freq = freq, periodogram = I))
}

log_likelihood_arctanh <- function(phi, sigma_eta, freq, periodogram) { 
  # phi <- tanh(params[1])
  # sigma_eta <- sqrt(exp(params[2]))
  
  llh <- 0
  # for (k in 1:length(freq)) {
    spec_dens_x <- sigma_eta^2/( 1 + phi^2 - 2 * phi * cos(2 * pi * freq))
    spec_dens_xi <- rep(pi^2/2, length(freq))
    spec_dens_y <- spec_dens_x + spec_dens_xi
    llh <- - log(spec_dens_y) - periodogram / spec_dens_y
  # }
  llh <- sum(llh)
}

likelihood_arctanh <- function(phi, sigma_eta, freq, periodogram) { 
  # phi <- tanh(params[1])
  # sigma_eta <- sqrt(exp(params[2]))
  
  llh <- 0
  # for (k in 1:length(freq)) {
  spec_dens_x <- sigma_eta^2/( 1 + phi^2 - 2 * phi * cos(2 * pi * freq))
  spec_dens_xi <- rep(pi^2/2, length(freq))
  spec_dens_y <- spec_dens_x + spec_dens_xi
  llh <- - log(spec_dens_y) - periodogram / spec_dens_y
  # }
  llh <- sum(llh)
  browser()
  likelihood <- exp(llh)

}

  
############################### Batch VB #########################

VB_iters <- 100

## Initial values
mu_0 <- prior_mean
gamma_0 <- c(log(sqrt(diag(prior_var))), sqrt(prior_var[lower.tri(prior_var)]))
## Construct mu and L out of lambda_0

lambda_0 <- list(mu = mu_0, gamma = gamma_0)
lambda <- list()
lambda[[1]] <- lambda_0

## Dimension variables set for convenience
p <- 3
d <- length(prior_mean)

## Set initial values for ADAM
m0 <- 0
v0 <- 0
n_variational_params <- d + d*(d+1)/2
m <- matrix(NA, nrow = n_variational_params, ncol = VB_iters+1)
v <- matrix(NA, nrow = n_variational_params, ncol = VB_iters+1)
m[, 1] <- rep(m0, n_variational_params)
v[, 1] <- rep(v0, n_variational_params)
t1 <- 0.9 #0.75
t2 <- 0.99 #0.9
alpha <- 0.005 ## Play around with the learning rate, increase it
eps <- 1e-08

## Pre-compute periodogram
fft_output <- compute_periodogram(y)

LB <- c()
# z <- rnorm(d, 0, 1)

# # Sample theta from q()
# L_0 <- diag(exp(gamma_0[1:length(prior_mean)])) + matrix(c(0, gamma_0[3], 0, 0), 2, 2) # change this later
# theta_0 <- mu_0 + L_0 %*% z

# params <- list(phi = tanh(theta_0[1]), sigma_eta = sqrt(exp(theta_0[2])))
# browser()
# log_like <- log_likelihood_arctanh(phi = params$phi, sigma_eta = params$sigma_eta, 
#                                     freq = fft_output$freq, periodogram = fft_output$periodogram)
# log_prior <- dmvnorm(unlist(params), prior_mean, prior_var, log = TRUE)
# LB[1] <- log_like + log_prior + sum(log(diag(L_0))) + 0.5 * t(L_0 %*% z) %*% chol2inv(L_0) %*% (L_0 %*% z)


for (s in 1:VB_iters) {
  cat("s =", s, "\n")

  mu_s <- lambda[[s]]$mu 
  gamma_s <- lambda[[s]]$gamma

  # ## Gradient
  z <- rnorm(d, 0, 1)

  # Sample theta from q()
  L_s <- diag(exp(gamma_s[1:d])) + matrix(c(0, gamma_s[d+1], 0, 0), 2, 2) # change this later
  theta <- lambda[[s]]$mu + L_s %*% z

  params <- list(phi = tanh(theta[1]), sigma_eta = sqrt(exp(theta[2])))
  
  ## Compute gradient

  grad_log_like_func <- Deriv(log_likelihood_arctanh, c("phi", "sigma_eta"))#compute_whittle_likelihood_sv(y, params)
  grad_log_like <- grad_log_like_func(phi = params$phi, 
                                sigma_eta = params$sigma_eta, 
                                freq = fft_output$freq, 
                                periodogram = fft_output$periodogram)

  grad_log_prior <- as.vector(solve(prior_var, prior_mean - theta))
  grad_log_h <- grad_log_like + grad_log_prior

  grad_LB_mu <- grad_log_h
  
  # test <- likelihood_arctanh(phi = params$phi, 
  #                            sigma_eta = params$sigma_eta, 
  #                            freq = fft_output$freq, 
  #                            periodogram = fft_output$periodogram)
  # browser()
  # grad_like_func <- Deriv(likelihood_arctanh, c("phi", "sigma_eta"))
  # grad_like <- grad_like_func(phi = params$phi, 
  #                       sigma_eta = params$sigma_eta, 
  #                       freq = fft_output$freq, 
  #                       periodogram = fft_output$periodogram)
  # browser()
  # k <- 2
  # inv_prior_var <- 1/prior_var # since the prior var is diagonal
  # grad_prior <- (2*pi)^(k/2) * sqrt(det(prior_var)) * 
  #   exp(-0.5*t(theta - prior_mean) %*% inv_prior_var %*% (theta - prior_mean)) *
  #   (-0.5 * inv_prior_var %*% (theta - prior_mean))
                  
  grad_LB_L <- grad_log_h %*% t(z) + diag(1/(diag(L_s)))
  
  # grad_LB_L <- grad_log_h %*% t(z) + chol2inv(L_s) %*% L_s %*% z %*% t(z)


  inds <- data.frame(i = c(1,2,1), j = c(1,2,2))
  grad_LB_gamma <- c()
  for (r in 1:nrow(inds)) {
    ind <- as.numeric(inds[r, ])
    i <- ind[1]
    j <- ind[2]
    grad_L_gamma_ij <- matrix(0, 2, 2)
    if (i == j) {
      grad_L_gamma_ij[i,j] <- exp(gamma_s[r])
    } else {
      grad_L_gamma_ij[i,j] <- 1
    }
    grad_LB_gamma[r] <- sum(diag(t(grad_LB_L) %*% grad_L_gamma_ij))
  }

  grad_LB <- c(grad_LB_mu, grad_LB_gamma)

  ## Compute LB
  log_like <- log_likelihood_arctanh(phi = params$phi, sigma_eta = params$sigma_eta, 
                                    freq = fft_output$freq, periodogram = fft_output$periodogram)
  log_prior <- dmvnorm(t(theta), prior_mean, prior_var, log = TRUE)
  
  LB[s] <- log_like + log_prior - dmvnorm(as.vector(theta), lambda[[s]]$mu, tcrossprod(L_s), log = T)

  ## Learning rate 
  ## Set learning rate and update parameters
  if (use_adam) {
    m[, s+1] <- t1 * m[, s] + (1 - t1) * grad_LB
    v[, s+1] <- t2 * v[, s] + (1 - t2) * grad_LB^2
    m_hat_s <- m[, s+1] / (1 - t1^(s+1))
    v_hat_s <- v[, s+1] / (1 - t2^(s+1))
    
    delta <- (alpha * m_hat_s) / (sqrt(v_hat_s) + eps)
    
    # lambda_new <- unlist(lambda) + delta
    lambda_new <- unlist(lambda[[s]], use.names = F) + delta
  } else {
    a_s <- 1/(1000+s)
    lambda_new <- unlist(lambda[[s]], use.names = F) + a_s * grad_LB
  }

  ## Update
  
  lambda[[s+1]] <- list(mu = lambda_new[1:d], gamma = lambda_new[(d+1):length(lambda_new)])

}


## Final estimates
vb.phi <- tanh(lambda[[VB_iters]]$mu[1])
vb.sigma_eta <- sqrt(exp(lambda[[VB_iters]]$mu[2]))

plot(LB, type = "l")
