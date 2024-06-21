## Stochastic volatility model
setwd("~/R-VGA-Whittle/02_SV/")

rm(list = ls())

library(mvtnorm)
library(coda)
library(Deriv)
library(cmdstanr)
library(tensorflow)
reticulate::use_condaenv("myenv", required = TRUE)
library(keras)
library(stats)
library(bspec)
library(tidyr)
library(ggplot2)
library(grid)
library(gridExtra)
library(gtable)

source("./source/compute_whittle_likelihood_sv.R")
# source("./source/run_rvgaw_sv_tf.R")
source("./source/run_rvgaw_sv_block.R")

source("./source/run_mcmc_sv.R")
source("./source/run_hmc_sv.R")
source("./source/compute_periodogram.R")
source("./source/find_cutoff_freq.R")

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

date <- "20240214" #"20230918" #the 20230918 version has sigma_eta = sqrt(0.1)
# date <- "20230918"

## R-VGA flags
regenerate_data <- F
save_data <- F
use_tempering <- T
temper_first <- T
reorder <- 0 #"decreasing" # or decreasing # or a number
reorder_seed <- 2024
plot_prior <- F
plot_likelihood_surface <- F
prior_type <- ""
transform <- "arctanh"
plot_trajectories <- T

n_post_samples <- 10000 # per chain 
burn_in <- 5000 # per chain
n_chains <- 2

## Flags
rerun_rvgaw <- F
rerun_mcmcw <- F
rerun_hmc <- F
rerun_hmcw <- F

save_rvgaw_results <- F
save_mcmcw_results <- F
save_hmc_results <- F
save_hmcw_results <- F

## Result directory
# result_directory <- paste0("./results/", prior_type, "/")
result_directory <- paste0("./results/", transform, "/")

## Generate data
mu <- 0
phi <- 0.99
sigma_eta <- 0.1 #sqrt(0.1)
sigma_eps <- 1
kappa <- 2
n <- 10000

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

## Generate data
if (regenerate_data) {
  set.seed(2023)
  x <- c()
  
  x1 <- rnorm(1, mu, sigma_eta^2 / (1 - phi^2))
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


## Test likelihood computation
if (plot_likelihood_surface) {
  param_grid <- seq(0.01, 0.99, length.out = 100)
  llh1 <- c() # log likelihood when varying phi
  llh2 <- c() # log likelihood when varying sigma_eta
  
  for (k in 1:length(param_grid)) {
    params1 <- list(phi = param_grid[k], sigma_eta = sigma_eta, sigma_xi = sqrt(pi^2/2))
    params2 <- list(phi = phi, sigma_eta = param_grid[k], sigma_xi = sqrt(pi^2/2))
    
    llh1[k] <- compute_whittle_likelihood_sv(y = y, params = params1)
    llh2[k] <- compute_whittle_likelihood_sv(y = y, params = params2)
    
  }
  
  par(mfrow = c(1,2))
  plot_range <- 1:100
  plot(param_grid[plot_range], llh1[plot_range], type = "l", 
       xlab = "Parameter", ylab = "Log likelihood", main = "phi")
  abline(v = phi, lty = 2)
  abline(v = param_grid[which.max(llh1)], col = "red", lty = 2)
  
  plot_range <- 1:100
  plot(param_grid[plot_range], llh2[plot_range], type = "l", 
       xlab = "Parameter", ylab = "Log likelihood", main = "sigma_eta")
  abline(v = sigma_eta, lty = 2)
  abline(v = param_grid[which.max(llh2)], col = "red", lty = 2)

}



########################################
##                Prior               ##
########################################

if (prior_type == "prior1") {
  prior_mean <- c(0, -4) #rep(0,2)
  prior_var <- diag(c(0.5, 0.5)) #diag(1, 2)
} else {
  prior_mean <- c(2, -3) #rep(0,2)
  prior_var <- diag(c(0.5, 0.5)) #diag(1, 2)
}

if (plot_prior) {
  prior_theta <- rmvnorm(10000, prior_mean, prior_var)
  prior_phi <- c()
  if (transform == "arctanh") {
    prior_phi <- tanh(prior_theta[, 1])
  } else {
    prior_phi <- exp(prior_theta[, 1]) / (1 + exp(prior_theta[, 1]))
    
  }
  prior_eta <- sqrt(exp(prior_theta[, 2]))
  # prior_xi <- sqrt(exp(prior_theta[, 3]))
  # par(mfrow = c(2,1))
  hist(prior_phi, main = "Prior of phi")
  hist(prior_eta, main = "Prior of sigma_eta")
}

########################################
##            R-VGA-Whitle            ##
########################################

S <- 1000L
# nblocks <- 100
blocksize <- 100
n_indiv <- find_cutoff_freq(y, nsegs = 25, power_prop = 1/2)$cutoff_ind #100

if (use_tempering) {
  n_temper <- 5
  K <- 100
  temper_schedule <- rep(1/K, K)
  temper_info <- ""
  if (temper_first) {
    temper_info <- paste0("_temperfirst", n_temper)
  } else {
    temper_info <- paste0("_temperlast", n_temper)
  }
  
} else {
  temper_info <- ""
}

if (reorder == "random") {
  reorder_info <- paste0("_", reorder, reorder_seed)
} else if (reorder == "decreasing") {
  reorder_info <- paste0("_", reorder)
} else if (reorder > 0) {
  reorder_info <- paste0("_reorder", reorder)
} else {
  reorder_info <- ""
}

# if (!is.null(nblocks)) {
if (!is.null(blocksize)) {
  # block_info <- paste0("_", nblocks, "blocks", n_indiv, "indiv")
  block_info <- paste0("_", "blocksize", blocksize, "_", n_indiv, "indiv")
} else {
  block_info <- ""
}

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_n", n, 
                         "_phi", phi_string, temper_info, reorder_info, block_info,
                         prior_type, "_", date, ".rds")

if (rerun_rvgaw) {
  rvgaw_results <- run_rvgaw_sv(y = y, #sigma_eta = sigma_eta, sigma_eps = sigma_eps, 
                                prior_mean = prior_mean, prior_var = prior_var, 
                                deriv = "tf", 
                                n_post_samples = (n_post_samples - burn_in) * n_chains,
                                S = S, use_tempering = use_tempering, 
                                temper_first = temper_first,
                                reorder = reorder,
                                n_temper = n_temper,
                                temper_schedule = temper_schedule, 
                                transform = transform,
                                # nblocks = nblocks,
                                blocksize = blocksize,
                                n_indiv = n_indiv)
  
  if (save_rvgaw_results) {
    saveRDS(rvgaw_results, rvgaw_filepath)
  }
  
} else {
  rvgaw_results <- readRDS(rvgaw_filepath)
}

rvgaw.post_samples_phi <- rvgaw_results$post_samples$phi
rvgaw.post_samples_sigma_eta <- rvgaw_results$post_samples$sigma_eta
# rvgaw.post_samples_xi <- rvgaw_results$post_samples$sigma_xi

###############################
##         HMC-exact         ##
###############################

hmc_filepath <- paste0(result_directory, "hmc_results_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

# n_chains <- 2
# hmc_iters <- n_post_samples / n_chains
# burn_in_per_chain <- burn_in / n_chains # per chain


if (rerun_hmc) {
  hmc_results <- run_hmc_sv(data = y, transform = transform,
                             prior_mean = prior_mean, prior_var = prior_var,
                             iters = n_post_samples, 
                             burn_in = burn_in,
                            n_chains = n_chains)
  
  if (save_hmc_results) {
    saveRDS(hmc_results, hmc_filepath)
  }
  
} else {
  hmc_results <- readRDS(hmc_filepath)
}

# hmc.fit <- extract(hfit, pars = c("theta_phi", "theta_sigma"),
#                    permuted = F)
# 
# hmc.theta_phi <- hmc.fit[,,1]
# hmc.theta_sigma <- hmc.fit[,,2]

hmc.post_samples_phi <- c(hmc_results$draws[,,1])#tanh(hmc.theta_phi)
hmc.post_samples_sigma_eta <- c(hmc_results$draws[,,2])#sqrt(exp(hmc.theta_sigma))

#####################################
##          HMC-Whittle            ##
#####################################
hmcw_filepath <- paste0(result_directory, "hmcw_results_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

if (rerun_hmcw) {
  
  # Compute periodogram
  y_tilde <- log(y^2) - mean(log(y^2))
  pgram_out <- compute_periodogram(y_tilde)
  freq <- pgram_out$freq
  I <- pgram_out$periodogram

  whittle_stan_file <- "./source/stan_sv_whittle.stan"
  # whittle_stan_file <- "./source/stan_mwe.stan" # this was to test the use of complex numbers

  whittle_sv_model <- cmdstan_model(
    whittle_stan_file,
    cpp_options = list(stan_threads = TRUE)
  )

  # log_kappa2_est <- mean(log(y^2)) - (digamma(1/2) + log(2))
  whittle_sv_data <- list(nfreq = length(freq), freqs = freq, periodogram = I,
                          prior_mean = prior_mean, diag_prior_var = diag(prior_var),
                          transform = ifelse(transform == "arctanh", 1, 0))

  # hfit <- stan(model_code = sv_code, 
  #              model_name="sv", data = sv_data, 
  #              iter = iters, warmup = burn_in, chains=1)

  fit_stan_multi_sv_whittle <- whittle_sv_model$sample(
    whittle_sv_data,
    chains = n_chains,
    threads = parallel::detectCores(),
    refresh = 100,
    iter_warmup = burn_in,
    iter_sampling = n_post_samples
  )

  hmcw_results <- list(draws = fit_stan_multi_sv_whittle$draws(variables = c("phi", "sigma_eta")),
                      time = fit_stan_multi_sv_whittle$time)

  if (save_hmcw_results) {
    saveRDS(hmcw_results, hmcw_filepath)
  }

} else {
  hmcw_results <- readRDS(hmcw_filepath)
}

# hmcw.fit <- extract(hfit, pars = c("theta_phi", "theta_sigma"),
#                    permuted = F)
# 
hmcw.post_samples_phi <- c(hmcw_results$draws[,,1])
hmcw.post_samples_sigma_eta <- c(hmcw_results$draws[,,2])

# ## Timing comparison
# rvgaw.time <- rvgaw_results$time_elapsed[3]
# hmcw.time <- sum(hmcw_results$time()$chains$total)
# hmc.time <- sum(hmc_results$time()$chains$total)
# print(data.frame(
#     method = c("R-VGA-Whittle", "HMC-Whittle", "HMC-exact"),
#     time = c(rvgaw.time, hmcw.time, hmc.time)
# ))