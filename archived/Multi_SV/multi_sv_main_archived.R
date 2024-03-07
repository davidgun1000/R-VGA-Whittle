## R-VGAW on the SV model

## MCMC on multivariate SV model
rm(list = ls())
setwd("~/R-VGA-Whittle/Multi_SV/")

library("coda")
library("mvtnorm")
library("astsa")
# library("expm")
library("stcos")
library(tensorflow)
reticulate::use_condaenv("tf2.11", required = TRUE)
library(keras)

source("./archived/run_rvgaw_multi_sv_archived.R")
source("./archived/run_mcmc_multi_sv_archived.R")
source("./archived/compute_whittle_likelihood_multi_sv.R")
source("./archived/construct_prior.R")
source("./archived/map_functions.R")
# source("./archived/compute_partial_whittle_likelihood.R")
source("./archived/compute_grad_hessian.R")

# List physical devices
gpus <- tf$config$experimental$list_physical_devices('GPU')

if (length(gpus) > 0) {
  tryCatch({
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    tf$config$experimental$set_virtual_device_configuration(
      gpus[[1]],
      list(tf$config$experimental$VirtualDeviceConfiguration(memory_limit=4096))
    )
    
    logical_gpus <- tf$config$experimental$list_logical_devices('GPU')
    
    print(paste0(length(gpus), " Physical GPUs,", length(logical_gpus), " Logical GPUs"))
  }, error = function(e) {
    # Virtual devices must be set before GPUs have been initialized
    print(e)
  })
}

## Flags
use_tempering <- F
reorder_freq <- T
decreasing <- T

#######################
##   Generate data   ##
#######################

phi11 <- 0.9
phi12 <- 0.1
phi21 <- 0.2
phi22 <- 0.7

Phi <- matrix(c(phi11, phi12, phi21, phi22), 2, 2, byrow = T)

sigma_eta1 <- 0.5#4
sigma_eta2 <- 0.6#7
Sigma_eta <- diag(c(sigma_eta1, sigma_eta2))

sigma_eps1 <- 1 #0.01
sigma_eps2 <- 1 #0.02

Sigma_eps <- diag(c(sigma_eps1, sigma_eps2))

Tfin <- 1000
x1 <- c(0, 0)
X <- matrix(NA, nrow = length(x1), ncol = Tfin+1) # x_0:T
X[, 1] <- x1
Y <- matrix(NA, nrow = length(x1), ncol = Tfin) # y_1:T
# Y[, 1] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
set.seed(2023)
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

################################
##    R-VGAW implementation   ##
################################

## Construct initial distribution/prior
prior <- construct_prior(data = Y)
prior_mean <- prior$prior_mean
prior_var <- prior$prior_var

param_dim <- length(prior_mean)

S <- 200L
a_vals <- 1
################ R-VGA starts here #################
print("Starting R-VGAL with Whittle likelihood...")

rvgaw_results <- run_rvgaw_multi_sv(data = Y, prior_mean = prior_mean, 
                                    prior_var = prior_var, S = S,
                                    use_tempering = use_tempering, 
                                    temper_schedule = temper_schedule, 
                                    reorder_freq = reorder_freq, decreasing = decreasing)

rvgaw.post_samples_Phi <- rvgaw_results$post_samples$Phi
rvgaw.post_samples_Sigma_eta <- rvgaw_results$post_samples$Sigma_eta

rvgaw.post_samples_phi_11 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[1,1]))
rvgaw.post_samples_phi_12 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[1,2]))
rvgaw.post_samples_phi_21 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[2,1]))
rvgaw.post_samples_phi_22 <- unlist(lapply(rvgaw.post_samples_Phi, function(x) x[2,2]))

rvgaw.post_samples_sigma_eta_11 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[1,1]))
rvgaw.post_samples_sigma_eta_12 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[1,2]))
rvgaw.post_samples_sigma_eta_21 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[2,1]))
rvgaw.post_samples_sigma_eta_22 <- unlist(lapply(rvgaw.post_samples_Sigma_eta, function(x) x[2,2]))

par(mfrow = c(2,2))
plot(density(rvgaw.post_samples_phi_11), col = "blue", main = "phi_11")
abline(v = Phi[1,1], lty = 2)
plot(density(rvgaw.post_samples_phi_12), col = "blue", main = "phi_12")
abline(v = Phi[1,2], lty = 2)
plot(density(rvgaw.post_samples_phi_21), col = "blue", main = "phi_21")
abline(v = Phi[2,1], lty = 2)
plot(density(rvgaw.post_samples_phi_22), col = "blue", main = "phi_22")
abline(v = Phi[2,2], lty = 2)

plot(density(rvgaw.post_samples_sigma_eta_11), col = "blue", main = "sigma_eta_11")
abline(v = Sigma_eta[1,1], lty = 2)
plot(density(rvgaw.post_samples_sigma_eta_12), col = "blue", main = "sigma_eta_12")
abline(v = Sigma_eta[1,2], lty = 2)
plot(density(rvgaw.post_samples_sigma_eta_21), col = "blue", main = "sigma_eta_21")
abline(v = Sigma_eta[2,1], lty = 2)
plot(density(rvgaw.post_samples_sigma_eta_22), col = "blue", main = "sigma_eta_22")
abline(v = Sigma_eta[2,2], lty = 2)

#############################
##   MCMC implementation   ##
#############################

n_post_samples <- 10000
burn_in <- 5000
iters <- n_post_samples + burn_in

mcmcw_results <- run_mcmc_multi_sv(data = Y, iters = iters, burn_in = burn_in, 
                                   prior_mean = prior_mean, prior_var = prior_var,
                                   adapt_proposal = T, use_whittle_likelihood = T)

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


par(mfrow = c(4,2))
traceplot(mcmcw.post_samples_phi_11, main = "Trace plot for phi_11")
abline(h = phi11, col = "red", lty = 2)
traceplot(mcmcw.post_samples_phi_12, main = "Trace plot for phi_12")
abline(h = phi12, col = "red", lty = 2)
traceplot(mcmcw.post_samples_phi_21, main = "Trace plot for phi_21")
abline(h = phi21, col = "red", lty = 2)
traceplot(mcmcw.post_samples_phi_22, main = "Trace plot for phi_22")
abline(h = phi22, col = "red", lty = 2)

traceplot(mcmcw.post_samples_sigma_eta_11, main = "Trace plot for sigma_eta_11")
abline(h = Sigma_eta[1,1], col = "red", lty = 2)
traceplot(mcmcw.post_samples_sigma_eta_12, main = "Trace plot for sigma_eta_12")
abline(h = Sigma_eta[1,2], col = "red", lty = 2)
traceplot(mcmcw.post_samples_sigma_eta_21, main = "Trace plot for sigma_eta_21")
abline(h = Sigma_eta[2,1], col = "red", lty = 2)
traceplot(mcmcw.post_samples_sigma_eta_22, main = "Trace plot for sigma_eta_22")
abline(h = Sigma_eta[2,2], col = "red", lty = 2)

par(mfrow = c(2,4))
plot(density(mcmcw.post_samples_phi_11), col = "blue", main = "phi_11")
lines(density(rvgaw.post_samples_phi_11), col = "red")
abline(v = phi11, lty = 2)

plot(density(mcmcw.post_samples_phi_12), col = "blue", main = "phi_12")
lines(density(rvgaw.post_samples_phi_12), col = "red")
abline(v = phi12, lty = 2)

plot(density(mcmcw.post_samples_phi_21), col = "blue", main = "phi_21")
lines(density(rvgaw.post_samples_phi_21), col = "red")
abline(v = phi21, lty = 2)

plot(density(mcmcw.post_samples_phi_22), col = "blue", main = "phi_22")
lines(density(rvgaw.post_samples_phi_22), col = "red")
abline(v = phi22, lty = 2)

plot(density(mcmcw.post_samples_sigma_eta_11), col = "blue", main = "sigma_eta_11")
lines(density(rvgaw.post_samples_sigma_eta_11), col = "red")
abline(v = Sigma_eta[1,1], lty = 2)

plot(density(mcmcw.post_samples_sigma_eta_12), col = "blue", main = "sigma_eta_12")
lines(density(rvgaw.post_samples_sigma_eta_12), col = "red")
abline(v = Sigma_eta[1,2], lty = 2)

plot(density(mcmcw.post_samples_sigma_eta_21), col = "blue", main = "sigma_eta_21")
lines(density(rvgaw.post_samples_sigma_eta_21), col = "red")
abline(v = Sigma_eta[2,1], lty = 2)

plot(density(mcmcw.post_samples_sigma_eta_22), col = "blue", main = "sigma_eta_22")
lines(density(rvgaw.post_samples_sigma_eta_22), col = "red")
abline(v = Sigma_eta[2,2], lty = 2)


