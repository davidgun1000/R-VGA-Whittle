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
# source("./archived/compute_whittle_likelihood_multi_sv.R")
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

phi11 <- 0.7#0.9
phi12 <- 0.1
phi21 <- 0.2
phi22 <- 0.85 #0.7

Phi <- matrix(c(phi11, phi12, phi21, phi22), 2, 2, byrow = T)

sigma_eta1 <- 0.4
sigma_eta2 <- 0.7
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

# rvgaw.t1 <- proc.time()
# 
# 
# rvgaw.mu_vals <- list()
# rvgaw.mu_vals[[1]] <- prior_mean
# 
# rvgaw.prec <- list()
# rvgaw.prec[[1]] <- chol2inv(chol(prior_var))
# 
# ## Generate a bunch of samples from the initial distribution
# ## Initial values: sample params from prior
# samples <- rmvnorm(S, prior_mean, prior_var)
# 
# # ## Construct A and Sigma_eta from these prior samples
# # ## Maybe this should be done in tensorflow?
# # samples2 <- lapply(seq_len(nrow(samples)), function(i) samples[i,])
# # 
# # ### the first 4 elements will be used to construct A
# # A_samples <- lapply(samples2, function(x) matrix(x[1:4], 2, 2, byrow = T))
# # 
# # ### the last 3 will be used to construct L
# # construct_Sigma_eta <- function(theta) {
# #   L <- diag(exp(theta[5:6]))
# #   L[2,1] <- theta[7]
# #   Sigma_eta <- L %*% t(L)
# #   return(Sigma_eta)
# # }
# # 
# # # L <- diag(exp(theta_samples[1, 5:6]))
# # # L[2,1] <- theta_samples[1, 7]
# # # Sigma_eta_curr <- L %*% t(L)
# # 
# # Sigma_eta_samples <- lapply(samples2, construct_Sigma_eta)
# # 
# # ## Transform samples of A into samples of Phi via the mapping in Ansley and Kohn (1986)
# # Phi_samples <- mapply(backward_map, A_samples, Sigma_eta_samples, SIMPLIFY = F)
# # 
# # j <- 2
# # llh <- list()
# # for (s in 1:S) {
# #   params_s <- list(Phi = Phi_samples[[s]], Sigma_eta = Sigma_eta_samples[[s]])
# #   llh[[s]] <- compute_partial_whittle_likelihood(Y = Y, params = params_s, j = j)
# # }
# 
# # ## Calculation of Whittle likelihood
# ## Fourier frequencies
# k <- seq(-ceiling(Tfin/2)+1, floor(Tfin/2), 1)
# k_in_likelihood <- k [k >= 1 & k <= floor((Tfin-1)/2)]
# freq <- 2 * pi * k_in_likelihood / Tfin
# 
# # ## astsa package
# Z <- log(Y^2) - rowMeans(log(Y^2))
# fft_out <- mvspec(t(Z), plot = F)
# I <- fft_out$fxx
# 
# if (reorder_freq) { # randomise order of frequencies and periodogram components
#   
#   if (decreasing) {
#     sorted_freq <- sort(freq, decreasing = T, index.return = T)
#     indices <- sorted_freq$ix
#     reordered_freq <- sorted_freq$x
#     reordered_I <- I[,,indices]
#   } else {
#     set.seed(reorder_seed)
#     indices <- sample(1:length(freq), length(freq))
#     reordered_freq <- freq[indices]
#     reordered_I <- I[,,indices]
#   }
#   
#   # plot(reordered_freq, type = "l")
#   # lines(freq, col = "red")
#   freq <- reordered_freq
#   I <- reordered_I 
# }
# 
# #### TF starts ##########
# # j <- 1
# # samples_tf <- tf$Variable(samples)
# # I_tf <- tf$Variable(I_all[,,j])
# # freq_tf <- tf$Variable(freq[j])
# # 
# # llh_test <- compute_grad_hessian(samples_tf, I_i = I_tf, freq_i = freq_tf)
# 
# ### the last 3 will be used to construct L
# construct_Sigma_eta <- function(theta) {
#   L <- diag(exp(theta[5:6]))
#   L[2,1] <- theta[7]
#   Sigma_eta <- L %*% t(L)
#   return(Sigma_eta)
# }
# 
# # Find the expected grad and expected Hessian
# for (i in 1:length(freq)) {
#   
#   # cat("i =", i, "\n")
#   a_vals <- 1
#   if (use_tempering) {
#     if (i <= n_temper) { # only temper the first n_temper observations
#       a_vals <- temper_schedule
#     }
#   }
#   
#   mu_temp <- rvgaw.mu_vals[[i]]
#   prec_temp <- rvgaw.prec[[i]]
#   
#   # grads <- list()
#   # hessian <- list()
#   
#   E_grad <- 0
#   E_hessian <- 0
#   
#   for (v in 1:length(a_vals)) { # for each step in the tempering schedule
#     
#     a <- a_vals[v]
#     
#     P <- chol2inv(chol(prec_temp))
#     samples <- rmvnorm(S, mu_temp, P)
#     # theta_phi <- samples[, 1]
#     # theta_eta <- samples[, 2]
#     # # theta_xi <- samples[, 3]
#     # theta_xi <- log(pi^2/2)
#     
#     samples_tf <- tf$Variable(samples)
#     I_tf <- tf$Variable(I[,,i])
#     freq_tf <- tf$Variable(freq[i])
#     
#     tf_out <- compute_grad_hessian(samples_tf, I_i = I_tf, freq_i = freq_tf)
#     
#     # llh <- list()
#     # for (s in 1:S) {
#     #   
#     #   ### the first 4 elements will be used to construct A
#     #   A_s <- matrix(samples[s, 1:4], 2, 2, byrow = T)
#     #   
#     #   Sigma_eta_s <- construct_Sigma_eta(samples[s, ])
#     #   
#     #   ## Transform samples of A into samples of Phi via the mapping in Ansley and Kohn (1986)
#     #   Phi_s <- backward_map(A_s, Sigma_eta_s)
#     #   
#     #   llh[[s]] <- compute_partial_whittle_likelihood(Y, params = list(Phi = Phi_s,
#     #                                                                   Sigma_eta = Sigma_eta_s),
#     #                                                  j = indices[i])
#     # }
#     
#     grads_tf <- tf_out$grad
#     hessians_tf <- tf_out$hessian
#     E_grad_tf <- tf$reduce_mean(grads_tf, 0L)
#     E_hessian_tf <- tf$reduce_mean(hessians_tf, 0L)
#     
#     E_grad <- E_grad_tf
#     E_hessian <- E_hessian_tf
#     prec_temp <- prec_temp - a * E_hessian
#     
#     if(any(eigen(prec_temp)$value < 0)) {
#       browser()
#     }
#     
#     mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * as.matrix(E_grad))
#     # mu_temp <- mu_temp + 1/prec_temp * (a * E_grad)
#     
#   }  
#   
#   rvgaw.prec[[i+1]] <- prec_temp
#   rvgaw.mu_vals[[i+1]] <- mu_temp
#   
#   if (i %% floor(length(freq)/10) == 0) {
#     cat(floor(i/length(freq) * 100), "% complete \n")
#   }
#   
# }
# rvgaw.t2 <- proc.time()
# 
# ## Posterior samples
# rvgaw.post_var <- chol2inv(chol(rvgaw.prec[[length(freq)]]))
# 
# rvgaw.post_samples <- rmvnorm(10000, rvgaw.mu_vals[[length(freq)]], rvgaw.post_var) # these are samples of beta, log(sigma_a^2), log(sigma_e^2)
# 
# ## Construct A and Sigma_eta from these posterior samples
# ## Maybe this should be done in tensorflow?
# rvgaw.post_samples2 <- lapply(seq_len(nrow(rvgaw.post_samples)), function(i) rvgaw.post_samples[i,])
# 
# ### the first 4 elements will be used to construct A
# post_samples_A <- lapply(rvgaw.post_samples2, function(x) matrix(x[1:4], 2, 2, byrow = T))
# 
# ### the last 3 will be used to construct L
# construct_Sigma_eta <- function(theta) {
#   L <- diag(exp(theta[5:6]))
#   L[2,1] <- theta[7]
#   Sigma_eta <- L %*% t(L)
#   return(Sigma_eta)
# }
# 
# # L <- diag(exp(theta_samples[1, 5:6]))
# # L[2,1] <- theta_samples[1, 7]
# # Sigma_eta_curr <- L %*% t(L)
# 
# post_samples_Sigma_eta <- lapply(rvgaw.post_samples2, construct_Sigma_eta)
# 
# ## Transform samples of A into samples of Phi via the mapping in Ansley and Kohn (1986)
# post_samples_Phi <- mapply(backward_map, post_samples_A, post_samples_Sigma_eta, SIMPLIFY = F)

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


# ## Save results
# rvgaw_results <- list(mu = rvgaw.mu_vals,
#                       prec = rvgaw.prec,
#                       post_samples = rvgaw.post_samples,
#                       transform = transform,
#                       S = S,
#                       use_tempering = use_tempering,
#                       temper_schedule = a_vals,
#                       time_elapsed = rvgaw.t2 - rvgaw.t1)