setwd("~/R-VGA-Whittle/01_LGSS/")
rm(list = ls())

library(mvtnorm)
library(coda)
# library(Deriv)
library(tensorflow)
library(cmdstanr)
reticulate::use_condaenv("myenv", required = TRUE)
library(keras)
library(Matrix)
library(tidyr)
library(ggplot2)
library(grid)
library(gtable)
library(gridExtra)

# source("./source/run_rvgaw_lgss_tf.R")
source("./source/run_rvgaw_lgss_block.R")
source("./source/run_mcmc_lgss_allparams.R")
source("./source/compute_periodogram.R")
source("./source/compute_kf_likelihood.R")
source("./source/compute_whittle_likelihood_lgss.R")
source("./source/update_sigma.R")
source("./source/run_hmc_lgss.R")
source("./source/find_cutoff_freq.R")

################## Some code to limit tensorflow memory usage ##################

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

################## End of code to limit tensorflow memory usage ##################

## Result directory
result_directory <- "./results/"

## Flags
date <- "20230525"
# date <- "20230302"

regenerate_data <- F
save_data <- F

rerun_rvgaw <- F
rerun_mcmcw <- F
rerun_hmc <- F
rerun_hmcw <- F

save_rvgaw_results <- F
save_mcmcw_results <- F
save_hmc_results <- F
save_hmcw_results <- F

plot_likelihood_surface <- F
plot_trajectories <- F
save_plots <- F

## R-VGA flags
use_tempering <- T
temper_first <- T
reorder <- 0 #"decreasing"
reorder_seed <- 2024
# decreasing <- T
transform <- "arctanh"

## MCMC flags
adapt_proposal <- T
n_post_samples <- 20000
burn_in <- 10000

## True parameters
sigma_eps <- 0.5 # measurement error var
sigma_eta <- 0.7 # process error var
phi <- 0.9

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

n <- 10000

if (regenerate_data) {
  print("Generating data...")
  
  # Generate true process x_0:T
  x <- c()
  set.seed(2023)
  x0 <- rnorm(1, 0, sqrt(sigma_eta^2 / (1-phi^2)))
  x[1] <- x0
  for (t in 2:(n+1)) {
    x[t] <- phi * x[t-1] + rnorm(1, 0, sigma_eta)
  }
  
  # Generate observations y_1:T
  y <- x[2:(n+1)] + rnorm(n, 0, sigma_eps)
  
  ## Plot true process and observations
  # par(mfrow = c(1, 1))
  # plot(x, type = "l", main = "True process")
  # points(y, col = "cyan")
  
  lgss_data <- list(x = x, y = y, phi = phi, sigma_eps = sigma_eps, sigma_eta = sigma_eta)
  if (save_data) {
    saveRDS(lgss_data, file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  }
} else {
  print("Reading saved data...")
  lgss_data <- readRDS(file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  
}

y <- lgss_data$y
x <- lgss_data$x
phi <- lgss_data$phi
sigma_eps <- lgss_data$sigma_eps
sigma_eta <- lgss_data$sigma_eta

## MCMC settings
n_post_samples <- 20000
burn_in <- 10000
MCMC_iters <- n_post_samples + burn_in # Number of MCMC iterations

## Prior
prior_mean <- c(0, -1, -1)#rep(0, 3)
prior_var <- diag(c(1, 1, 1))

## Initial state mean and variance for the KF
# state_ini_mean <- 0
# state_ini_var <- 1

if (plot_likelihood_surface) {
## Test the likelihood computation by plotting likelihood surface over a grid of parameter values
phi_grid <- seq(-1, 1, length.out = 200)
likelihood_whittle <- c()
likelihood_exact <- c()

for (i in 1:length(phi_grid)) {
  params_list <- list(phi = phi_grid[i], sigma_eta = sigma_eta, sigma_eps = sigma_eps)
  # if (use_whittle_likelihood) {
  likelihood_whittle[i] <- compute_whittle_likelihood_lgss(y = y, params = params_list)
  # } else {
  kf_out <- compute_kf_likelihood(state_prior_mean = state_ini_mean,
                                  state_prior_var = state_ini_var,
                                  iters = length(y), observations = y,
                                  params = params_list)
  
  likelihood_exact[i] <- kf_out$log_likelihood
  # }
}
  par(mfrow = c(2,1))
  margin <- 20
  plot_range <- (which.max(likelihood_exact) - margin):(which.max(likelihood_exact) + margin)
  plot(phi_grid[plot_range], likelihood_exact[plot_range], type = "l",
      xlab = "phi", ylab = "log likelihood", main = paste0("Exact likelihood (n = ", n, ")"))
  legend("topleft", legend = c("true value", "arg max llh"),
        col = c("black", "red"), lty = 2, cex = 0.5)
  abline(v = phi_grid[which.max(likelihood_exact)], lty = 1, col = "red")
  abline(v = phi, lty = 2)

  plot(phi_grid[plot_range], likelihood_whittle[plot_range], type = "l",
      xlab = "phi", ylab = "log likelihood", main = paste0("Whittle likelihood (n = ", n, ")"))
  legend("topleft", legend = c("true value", "arg max llh"),
        col = c("black", "red"), lty = 2, cex = 0.5)
  abline(v = phi_grid[which.max(likelihood_whittle)], lty = 1, col = "red")
  abline(v = phi, lty = 2)

  browser()
}

## Sample from prior to see if values of phi are reasonable
# theta_samples <- rnorm(10000, prior_mean, prior_var)
# if (transform == "arctanh") {
#   phi_samples <- tanh(theta_samples)
# } else {
#   phi_samples <- exp(theta_samples) / (1 + exp(theta_samples))
# }
# 
# hist(phi_samples, main = "Samples from the prior of phi")

##########################################
##            R-VGA-Whittle             ##
##########################################
S <- 1000L

# nblocks <- 100
n_indiv <- find_cutoff_freq(y, nsegs = 25, power_prop = 1/2)$cutoff_ind #500 #220 #1000 #807
blocksize <- 100 #floor((n-1)/2) - n_indiv 

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

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_", transform, "_n", n,
                         "_phi", phi_string, temper_info, reorder_info, block_info, "_", date, ".rds")

if (rerun_rvgaw) {
  rvgaw_results <- run_rvgaw_lgss(y = y, #sigma_eta = sigma_eta, sigma_eps = sigma_eps, 
                                  prior_mean = prior_mean, prior_var = prior_var, 
                                  deriv = "tf", 
                                  S = S, n_post_samples = n_post_samples,
                                  use_tempering = use_tempering, 
                                  temper_first = temper_first,
                                  temper_schedule = temper_schedule,
                                  reorder = reorder,
                                  reorder_seed = reorder_seed,
                                  n_temper = n_temper,
                                  # nblocks = nblocks,
                                  blocksize = blocksize,
                                  n_indiv = n_indiv
                                  )
  
  if (save_rvgaw_results) {
    saveRDS(rvgaw_results, rvgaw_filepath)
  }
  
} else {
  rvgaw_results <- readRDS(rvgaw_filepath)
}

rvgaw.post_samples_phi <- rvgaw_results$post_samples$phi
rvgaw.post_samples_eta <- rvgaw_results$post_samples$sigma_eta
rvgaw.post_samples_eps <- rvgaw_results$post_samples$sigma_eps


#############################
##        HMC-exact        ##
#############################

hmc_filepath <- paste0(result_directory, "hmc_results_n", n, 
                       "_phi", phi_string, "_", date, ".rds")

# n_post_samples <- 10000
# burn_in <- 1000
# stan.iters <- n_post_samples + burn_in
n_chains <- 2
if (rerun_hmc) {
  hmc_results <- run_hmc_lgss(data = y, 
                                iters = n_post_samples / n_chains, 
                                burn_in = burn_in / n_chains)
  
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

hmc.post_samples_phi <- hmc_results$draws[,,1]#tanh(hmc.theta_phi)
hmc.post_samples_eta <- hmc_results$draws[,,2]#sqrt(exp(hmc.theta_sigma))
hmc.post_samples_eps <- hmc_results$draws[,,3]#sqrt(exp(hmc.theta_sigma))

####################################
##          HMC-Whittle           ##
####################################

hmcw_filepath <- paste0(result_directory, "hmcw_results_n", n, 
                       "_phi", phi_string, "_", date, ".rds")

if (rerun_hmcw) {
  
  # Compute periodogram
  pgram_output <- compute_periodogram(y)
  freq <- pgram_output$freq
  I <- pgram_output$periodogram
  
  whittle_stan_file <- "./source/stan_lgss_whittle.stan"
  
  whittle_lgss_model <- cmdstan_model(
    whittle_stan_file,
    cpp_options = list(stan_threads = TRUE)
  )
  
  # log_kappa2_est <- mean(log(y^2)) - (digamma(1/2) + log(2))
  
  whittle_lgss_data <- list(nfreq = length(freq), freqs = freq, periodogram = I)
  
  fit_stan_lgss_whittle <- whittle_lgss_model$sample(
    whittle_lgss_data,
    chains = n_chains,
    threads = parallel::detectCores(),
    refresh = 100,
    iter_warmup = burn_in / n_chains,
    iter_sampling = n_post_samples / n_chains
  )
  
  hmcw_results <- list(draws = fit_stan_lgss_whittle$draws(variables = c("phi", "sigma_eta", "sigma_eps")),
                       time = fit_stan_lgss_whittle$time,
                       summary = fit_stan_lgss_whittle$cmdstan_summary)
  # fit_stan_lgss_whittle$cmdstan_summary()
  # fit_stan_lgss_whittle$diagnostic_summary()
  
  if (save_hmcw_results) {
    saveRDS(hmcw_results, hmcw_filepath)
  }
  
} else {
  hmcw_results <- readRDS(hmcw_filepath)
}

hmcw.post_samples_phi <- c(hmcw_results$draws[,,1])
hmcw.post_samples_eta <- c(hmcw_results$draws[,,2])
hmcw.post_samples_eps <- c(hmcw_results$draws[,,3])


## Timing comparison
# rvgaw.time <- rvgaw_results$time_elapsed[3]
# hmcw.time <- sum(hmcw_results$time()$chains$total)
# hmc.time <- sum(hmc_results$time()$chains$total)
# print(data.frame(method = c("R-VGA", "HMCW", "HMC"),
#                  time = c(rvgaw.time, hmcw.time, hmc.time)))
