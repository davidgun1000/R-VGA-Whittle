## MCMC on multivariate SV model

rm(list = ls())
setwd("~/R-VGA-Whittle/Multi_SV/")

source("./archived/run_mcmc_multi_sv_archived.R")
source("./archived/compute_whittle_likelihood_multi_sv.R")
source("./archived/construct_prior.R")
source("./archived/map_functions.R")

library("coda")
library("mvtnorm")
library("astsa")
# library("expm")
library("stcos")

## Flags
use_whittle_likelihood <- T
adapt_proposal <- F

#######################
##   Generate data   ##
#######################

phi11 <- 0.7 #0.9
phi12 <- 0.3
phi21 <- 0.05
phi22 <- 0.85

Phi <- matrix(c(phi11, phi12, phi21, phi22), 2, 2, byrow = T)

sigma_eta1 <- 0.5
sigma_eta2 <- 0.6
Sigma_eta <- diag(c(sigma_eta1, sigma_eta2))

sigma_eps1 <- 1 #0.01
sigma_eps2 <- 1 #0.02

Sigma_eps <- diag(c(sigma_eps1, sigma_eps2))

Tfin <- 100
x1 <- c(0, 0)
X <- matrix(NA, nrow = length(x1), ncol = Tfin+1) # x_0:T
X[, 1] <- x1
Y <- matrix(NA, nrow = length(x1), ncol = Tfin) # y_1:T
# Y[, 1] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
for (t in 1:Tfin) {
  X[, t+1] <- Phi %*% X[, t] + t(rmvnorm(1, c(0, 0), Sigma_eta))
  V <- diag(exp(X[, t+1]/2))
  Y[, t] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
}

par(mfrow = c(2,1))
plot(X[1, ], type = "l")
plot(X[2, ], type = "l")

plot(Y[1, ], type = "l")
plot(Y[2, ], type = "l")

#############################
##   MCMC implementation   ##
#############################

n_post_samples <- 10000
burn_in <- 5000
iters <- n_post_samples + burn_in

# update_sigma <- function(sigma2, acc, p, i, d) { # function to adapt scale parameter in proposal covariance
#   alpha = -qnorm(p/2);
#   c = ((1-1/d)*sqrt(2*pi)*exp(alpha^2/2)/(2*alpha) + 1/(d*p*(1-p)));
#   Theta = log(sqrt(sigma2));
#   Theta = Theta+c*(acc-p)/max(200, i/d);
#   theta = (exp(Theta));
#   theta = theta^2;
#   
#   return(theta)
# }
# 
# mcmcw.t1 <- proc.time()
# 
# accept <- rep(0, iters)
# acceptProb <- c()
# 
# prior <- construct_prior(data = Y)
# prior_mean <- prior$prior_mean
# prior_var <- prior$prior_var
# 
# param_dim <- length(prior_mean)
# 
# post_samples <- list()
# post_samples_theta <- matrix(NA, iters, param_dim)
# 
# ## Initial values: sample params from prior
# theta_curr <- rmvnorm(1, prior_mean, prior_var)
# 
# ### the first 4 elements will be used to construct A
# A_curr <- matrix(theta_curr[1:4], 2, 2, byrow = T)
# 
# ### the last 3 will be used to construct L
# L <- diag(exp(theta_curr[5:6]))
# L[2,1] <- theta_curr[7]
# Sigma_eta_curr <- L %*% t(L)
# 
# ## 3. Map (A, Sigma_eta) to (Phi, Sigma_eta) using the mapping in Ansley and Kohn (1986)
# Phi_curr <- backward_map(A_curr, Sigma_eta_curr)
# params_curr <- list(Phi = Phi_curr, Sigma_eta = Sigma_eta_curr)
# 
# ## 4. Calculate the initial log likelihood
# if (use_whittle_likelihood) {
#   log_likelihood_curr <- compute_whittle_likelihood_multi_sv(Y = Y,
#                                                              params = params_curr)
# } else { 
#   # nothing here yet
# }
# 
# ## 5. Calculate the initial log prior
# log_prior_curr <- dmvnorm(theta_curr, prior_mean, prior_var, log = T)
# 
# ## Proposal variance
# D <- diag(rep(0.01, param_dim))
# if (adapt_proposal) {
#   scale <- 1
#   target_accept <- 0.23
# }
# 
# for (i in 1:iters) {
#   
#   # cat("i =", i, "\n")
#   if (i %% (iters/10) == 0) {
#     cat(i/iters * 100, "% complete \n")
#     cat("Acceptance rate:", sum(accept)/i, "\n")
#     # cat("Current params:", unlist(params_curr), "\n")
#     # cat("------------------------------------------------------------------\n")
#   }
#   
#   ## 1. Propose new parameter values
#   theta_prop <- rmvnorm(1, theta_curr, D)
#   
#   ### the first 4 elements will be used to construct A
#   A_prop <- matrix(theta_prop[1:4], 2, 2, byrow = T)
#   
#   ### the last 3 will be used to construct L
#   L <- diag(exp(theta_prop[5:6]))
#   L[2,1] <- theta_prop[7]
#   Sigma_eta_prop <- L %*% t(L)
#   
#   ## 3. Map (A, Sigma_eta) to (Phi, Sigma_eta) using the mapping in Ansley and Kohn (1986)
#   Phi_prop <- backward_map(A_prop, Sigma_eta_prop)
#   params_prop <- list(Phi = Phi_prop, Sigma_eta = Sigma_eta_prop)
#   
#   ## 2. Calculate likelihood
#   if (use_whittle_likelihood) {
#     log_likelihood_prop <- compute_whittle_likelihood_multi_sv(Y = Y,
#                                                                params = params_prop)
#   } else { 
#     # nothing here yet
#   }
#   
#   ## 3. Calculate prior
#   log_prior_prop <- dmvnorm(theta_prop, prior_mean, prior_var, log = T)
#   
#   # cat("llh_curr = ", log_likelihood_curr, " , llh_prop = ", log_likelihood_prop, "\n")
#   # cat("lprior_curr = ", log_prior_curr, " , lprior_prop = ", log_prior_prop, "\n")
#   
#   ## 4. Calculate acceptance probability
#   r <- exp((log_likelihood_prop + log_prior_prop) - 
#              (log_likelihood_curr + log_prior_curr))
#   # cat("r = ", r, "\n")
#   a <- min(1, r)
#   acceptProb[i] <- a
#   
#   ## 5. Move to next state with acceptance probability a
#   u <- runif(1, 0, 1)
#   if (u < a) {
#     accept[i] <- 1
#     theta_curr <- theta_prop # transformed parameters
#     params_curr <- params_prop # params on the original scale
#     log_likelihood_curr <- log_likelihood_prop
#     log_prior_curr <- log_prior_prop
#   } 
#   # cat("Acc =", accept[i], "\n")
#   
#   ## Store parameter 
#   # if (use_whittle_likelihood) {
#   post_samples_theta[i, ] <- theta_curr #c(params_curr$phi, params_curr$sigma_eta)
#   post_samples[[i]] <- list(Phi = params_curr$Phi, Sigma_eta = params_curr$Sigma_eta)
#   # } else {
#   #   post_samples_theta[i, ] <- c(params_curr$phi, params_curr$sigma_eta, params_curr$sigma_xi)
#   # }
#   
#   ## Adapt proposal covariance matrix
#   if (adapt_proposal) {
#     if ((i >= 100) && (i %% 10 == 0)) {
#       scale <- update_sigma(scale, a, target_accept, i, param_dim)
#       D <- scale * var((post_samples_theta[1:i, ]))
#     }
#   }
# }
# 
# mcmcw.t2 <- proc.time()

mcmcw_results <- run_mcmc_multi_sv(data = Y, iters = iters, burn_in = burn_in, 
                                  prior_mean = prior_mean, prior_var = prior_var,
                                  adapt_proposal = F, use_whittle_likelihood = T)

## Extract samples
mcmcw.post_samples_phi <- lapply(mcmcw_results$post_samples, function(x) x$Phi) #post_samples_theta[, 1]
mcmcw.post_samples_sigma_eta <- lapply(mcmcw_results$post_samples, function(x) x$Sigma_eta) #post_samples_theta[, 2]
# post_samples_xi <- post_samples_theta[, 3]
# mcmcw.post_samples <- list(phi = post_samples_phi,
#                           sigma_eta = post_samples_eta) #,
# sigma_xi = post_samples_xi)

mcmcw.post_samples_phi_11 <- lapply(mcmcw.post_samples_phi, function(x) x[1,1])
mcmcw.post_samples_phi_12 <- lapply(mcmcw.post_samples_phi, function(x) x[1,2])
mcmcw.post_samples_phi_21 <- lapply(mcmcw.post_samples_phi, function(x) x[2,1])
mcmcw.post_samples_phi_22 <- lapply(mcmcw.post_samples_phi, function(x) x[2,2])

mcmcw.post_samples_phi_11 <- as.mcmc(unlist(mcmcw.post_samples_phi_11[-(1:burn_in)]))
mcmcw.post_samples_phi_12 <- as.mcmc(unlist(mcmcw.post_samples_phi_12[-(1:burn_in)]))
mcmcw.post_samples_phi_21 <- as.mcmc(unlist(mcmcw.post_samples_phi_21[-(1:burn_in)]))
mcmcw.post_samples_phi_22 <- as.mcmc(unlist(mcmcw.post_samples_phi_22[-(1:burn_in)]))

mcmcw.post_samples_sigma_eta_11 <- lapply(mcmcw.post_samples_sigma_eta, function(x) x[1,1])
mcmcw.post_samples_sigma_eta_12 <- lapply(mcmcw.post_samples_sigma_eta, function(x) x[1,2])
mcmcw.post_samples_sigma_eta_21 <- lapply(mcmcw.post_samples_sigma_eta, function(x) x[2,1])
mcmcw.post_samples_sigma_eta_22 <- lapply(mcmcw.post_samples_sigma_eta, function(x) x[2,2])

mcmcw.post_samples_sigma_eta_11 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_11[-(1:burn_in)]))
mcmcw.post_samples_sigma_eta_12 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_12[-(1:burn_in)]))
mcmcw.post_samples_sigma_eta_21 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_21[-(1:burn_in)]))
mcmcw.post_samples_sigma_eta_22 <- as.mcmc(unlist(mcmcw.post_samples_sigma_eta_22[-(1:burn_in)]))


par(mfrow = c(2,2))
traceplot(mcmcw.post_samples_phi_11, main = "Trace plot for MCMC with exact likelihood")
abline(h = phi11, col = "red", lty = 2)
traceplot(mcmcw.post_samples_phi_12, main = "Trace plot for MCMC with exact likelihood")
abline(h = phi12, col = "red", lty = 2)
traceplot(mcmcw.post_samples_phi_21, main = "Trace plot for MCMC with exact likelihood")
abline(h = phi21, col = "red", lty = 2)
traceplot(mcmcw.post_samples_phi_22, main = "Trace plot for MCMC with exact likelihood")
abline(h = phi22, col = "red", lty = 2)

traceplot(mcmcw.post_samples_sigma_eta_11, main = "Trace plot for MCMC with exact likelihood")
abline(h = Sigma_eta[1,1], col = "red", lty = 2)
traceplot(mcmcw.post_samples_sigma_eta_12, main = "Trace plot for MCMC with exact likelihood")
abline(h = Sigma_eta[1,2], col = "red", lty = 2)
traceplot(mcmcw.post_samples_sigma_eta_21, main = "Trace plot for MCMC with exact likelihood")
abline(h = Sigma_eta[2,1], col = "red", lty = 2)
traceplot(mcmcw.post_samples_sigma_eta_22, main = "Trace plot for MCMC with exact likelihood")
abline(h = Sigma_eta[2,2], col = "red", lty = 2)


plot(density(mcmcw.post_samples_phi_11), col = "blue", main = "phi_11")
abline(v = phi11, lty = 2)
plot(density(mcmcw.post_samples_phi_12), col = "blue", main = "phi_12")
abline(v = phi12, lty = 2)
plot(density(mcmcw.post_samples_phi_21), col = "blue", main = "phi_21")
abline(v = phi21, lty = 2)
plot(density(mcmcw.post_samples_phi_22), col = "blue", main = "phi_22")
abline(v = phi22, lty = 2)

plot(density(mcmcw.post_samples_sigma_eta_11), col = "blue", main = "sigma_eta_11")
abline(v = Sigma_eta[1,1], lty = 2)
plot(density(mcmcw.post_samples_sigma_eta_12), col = "blue", main = "sigma_eta_12")
abline(v = Sigma_eta[1,2], lty = 2)
plot(density(mcmcw.post_samples_sigma_eta_21), col = "blue", main = "sigma_eta_21")
abline(v = Sigma_eta[2,1], lty = 2)
plot(density(mcmcw.post_samples_sigma_eta_22), col = "blue", main = "sigma_eta_22")
abline(v = Sigma_eta[2,2], lty = 2)


