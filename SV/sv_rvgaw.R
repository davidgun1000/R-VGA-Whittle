## R-VGA on SV model

setwd("~/R-VGA-Whittle/SV/")

library(mvtnorm)
library(Deriv)
library(tensorflow)
reticulate::use_condaenv("tf2.11", required = TRUE)
library(keras)

source("./source/run_rvgaw_sv_tf.R")
# source("./source/compute_kf_likelihood.R")
source("./source/compute_whittle_likelihood_sv.R")
# source("./source/update_sigma.R")


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
date <- "20230525"
regenerate_data <- T
save_data <- T
use_tempering <- T
reorder_freq <- F
decreasing <- F
reorder_seed <- 2024

## Generate data
mu <- 0
sigma_eta <- 0.5
phi <- 0.9
x1 <- rnorm(1, mu, sigma_eta^2 / (1 - phi^2))
n <- 1000
x <- c()
x[1] <- x1

for (t in 2:n) {
  x[t] <- mu + phi * (x[t-1] - mu) + rnorm(1, 0, sigma_eta)
}

eps <- rnorm(n, 0, 1)
y <- exp(x/2) * eps

par(mfrow = c(2,1))
plot(y, type = "l")
plot(x, type = "l")

## R-VGA ##

S <- 200

if (use_tempering) {
  n_temper <- 10
  K <- 10
  temper_schedule <- rep(1/K, K)
}

## Prior
prior_mean <- rep(0, 3) #c(atanh(phi), log(sigma_eta^2), log(pi^2/2)) #rep(0, 3)
prior_var <- diag(c(1, 1, 1))

prior_theta <- rmvnorm(10000, prior_mean, prior_var)
prior_phi <- tanh(prior_theta[, 1])
prior_eta <- sqrt(exp(prior_theta[, 2]))
prior_xi <- sqrt(exp(prior_theta[, 3]))
par(mfrow = c(3,1))
hist(prior_phi)
hist(prior_eta)
hist(prior_xi)

rvgaw_results <- run_rvgaw_sv(y = y, #sigma_eta = sigma_eta, sigma_eps = sigma_eps, 
                              prior_mean = prior_mean, prior_var = prior_var, 
                              deriv = "tf", 
                              S = S, use_tempering = use_tempering, 
                              reorder_freq = reorder_freq,
                              decreasing = decreasing, 
                              n_temper = n_temper,
                              temper_schedule = temper_schedule)

# if (save_rvgaw_results) {
#   saveRDS(rvgaw_results, rvgaw_filepath)
# }

rvgaw.post_samples <- rvgaw_results$post_samples$phi
rvgaw.post_samples_phi <- rvgaw_results$post_samples$phi
rvgaw.post_samples_eta <- rvgaw_results$post_samples$sigma_eta
rvgaw.post_samples_xi <- rvgaw_results$post_samples$sigma_xi


par(mfrow = c(1,3))
plot(density(rvgaw.post_samples_phi), main = "Posterior of phi")
abline(v = phi, lty = 2)

plot(density(rvgaw.post_samples_eta), main = "Posterior of sigma_eta")
abline(v = sigma_eta, lty = 2)

plot(density(rvgaw.post_samples_xi), main = "Posterior of sigma_xi")
abline(v = sqrt(pi^2/2), lty = 2)

## Trajectories
mu_phi <- sapply(rvgaw_results$mu, function(x) x[1])
mu_eta <- sapply(rvgaw_results$mu, function(x) x[2])
mu_xi <- sapply(rvgaw_results$mu, function(x) x[3])

par(mfrow = c(1, 3))
plot(tanh(mu_phi), type = "l", main = "Trajectory of phi")
abline(h = phi, lty = 2)

plot(sqrt(exp(mu_eta)), type = "l", main = "Trajectory of sigma_eta")
abline(h = sigma_eta, lty = 2)

plot(sqrt(exp(mu_xi)), type = "l", main = "Trajectory of sigma_xi")
abline(h = sqrt(pi^2/2), lty = 2)

