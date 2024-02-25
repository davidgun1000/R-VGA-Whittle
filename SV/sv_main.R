## Stochastic volatility model
setwd("~/R-VGA-Whittle/SV/")

rm(list = ls())

library(mvtnorm)
library(coda)
library(Deriv)
# library(rstan)
library(cmdstanr)
library(tensorflow)
reticulate::use_condaenv("myenv", required = TRUE)
library(keras)
library(stats)
library(bspec)

source("./source/compute_whittle_likelihood_sv.R")
# source("./source/run_rvgaw_sv_tf.R")
source("./source/run_rvgaw_sv_tf.R")

source("./source/run_mcmc_sv.R")
source("./source/run_hmc_sv.R")

# source("./source/run_corr_pmmh_sv.R")
source("./source/particleFilter.R")

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
use_tempering <- F
temper_first <- F
reorder <- "random" # or decreasing # or a number
# reorder_freq <- T
# decreasing <- F
reorder_seed <- 2020 #2024
plot_prior <- T
plot_likelihood_surface <- F
prior_type <- ""
transform <- "arctanh"
plot_trajectories <- F
use_welch <- F

## Flags
rerun_rvgaw <- T
rerun_mcmcw <- F
# rerun_mcmce <- F
rerun_hmc <- F
rerun_hmcw <- F

save_rvgaw_results <- F
save_mcmcw_results <- F
# save_mcmce_results <- F
save_hmc_results <- F
save_hmcw_results <- F

save_plots <- F

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

## Welch's

# # load example data:
# data(sunspots)
# # compute and plot the "plain" spectrum:
# spec1 <- empiricalSpectrum(sunspots)
# plot(spec1$frequency, spec1$power, type="l")

# # plot Welch spectrum using segments of length 10 years:
# spec2 <- welchPSD(sunspots, seglength=10)
# lines(spec2$frequency, spec2$power, col="red")

# par(mfrow = c(2,1))
# plot(y, type = "l")
# plot(x, type = "l")



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

  browser()
}
# 
# test_eps <- rnorm(10000, 0, 1)
# test_xi <- log(test_eps^2)
# var(test_xi)

# # Test exact likelihood
# phi_grid <- seq(0.01, 0.99, length.out = 100)
# llh <- c()
# 
# for (k in 1:length(phi_grid)) {
#   params_pf <- list(phi = phi_grid[k], sigma_eta = sigma_eta, sigma_eps = sigma_eps,
#                     sigma_xi = sqrt(pi^2/2))
#   pf_out <- particleFilter(y = y, N = 100, iniState = 0, param = params_pf)
#   llh[k] <- pf_out$log_likelihood
# }
# 
# plot(phi_grid, llh, type = "l")
# abline(v = phi_grid[which.max(llh)], col = "red", lty = 2)
# 
# browser()

########################################
##                Prior               ##
########################################

if (prior_type == "prior1") {
  prior_mean <- c(0, -4) #rep(0,2)
  prior_var <- diag(c(1, 2)) #diag(1, 2)
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
##                R-VGA               ##
########################################

S <- 5000L

if (use_tempering) {
  n_temper <- 10
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

if (use_welch) {
  welch_info <- "_welch"
} else {
  welch_info <- ""
}

if (reorder == "random" | reorder == "decreasing") {
  reorder_info <- reorder
} else if (reorder > 0) {
  reorder_info <- paste0("_reorder", reorder)
} else {
  reorder_info <- ""
}

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_n", n, 
                         "_phi", phi_string, temper_info, reorder_info, 
                         prior_type, "_", date, ".rds")

if (rerun_rvgaw) {
  rvgaw_results <- run_rvgaw_sv(y = y, #sigma_eta = sigma_eta, sigma_eps = sigma_eps, 
                                prior_mean = prior_mean, prior_var = prior_var, 
                                deriv = "tf", 
                                S = S, use_tempering = use_tempering, 
                                temper_first = temper_first,
                                reorder = reorder,
                                n_temper = n_temper,
                                temper_schedule = temper_schedule, 
                                transform = transform)
                                # use_welch = use_welch)
  
  if (save_rvgaw_results) {
    saveRDS(rvgaw_results, rvgaw_filepath)
  }
  
} else {
  rvgaw_results <- readRDS(rvgaw_filepath)
}

rvgaw.post_samples_phi <- rvgaw_results$post_samples$phi
rvgaw.post_samples_eta <- rvgaw_results$post_samples$sigma_eta
# rvgaw.post_samples_xi <- rvgaw_results$post_samples$sigma_xi

########################################
##                MCMC                ## 
########################################

mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_results_n", n, 
                         "_phi", phi_string, prior_type, "_", date, ".rds")

adapt_proposal <- T

n_post_samples <- 10000
burn_in <- 1000
MCMC_iters <- n_post_samples + burn_in

# prior_mean <- rep(0, 3)
# prior_var <- diag(c(1, 1, 0.01))

# prior_samples <- rmvnorm(10000, prior_mean, prior_var)
# prior_samples_phi <- tanh(prior_samples[, 1])                                       
# hist(prior_samples_sigma_xi)

if (rerun_mcmcw) {
  mcmcw_results <- run_mcmc_sv(y, #sigma_eta, sigma_eps, 
                               iters = MCMC_iters, burn_in = burn_in,
                               prior_mean = prior_mean, prior_var = prior_var,  
                               state_ini_mean = state_ini_mean, state_ini_var = state_ini_var,
                               adapt_proposal = T, use_whittle_likelihood = T,
                               transform = transform)
  
  if (save_mcmcw_results) {
    saveRDS(mcmcw_results, mcmcw_filepath)
  }
} else {
  mcmcw_results <- readRDS(mcmcw_filepath)
}

mcmcw.post_samples_phi <- as.mcmc(mcmcw_results$post_samples$phi[-(1:burn_in)])
mcmcw.post_samples_eta <- as.mcmc(mcmcw_results$post_samples$sigma_eta[-(1:burn_in)])
# mcmcw.post_samples_xi <- as.mcmc(mcmcw_results$post_samples$sigma_xi[-(1:burn_in)])

# par(mfrow = c(2,1))
# coda::traceplot(mcmcw.post_samples_phi, main = "Trace plot for phi")
# coda::traceplot(mcmcw.post_samples_eta, main = "Trace plot for sigma_eta")
# # traceplot(mcmcw.post_samples_xi, main = "Trace plot for sigma_xi")
# 
# par(mfrow = c(1,2))
# plot(density(mcmcw.post_samples_phi), main = "Posterior of phi", 
#      col = "blue", lty = 2, lwd = 3)
# lines(density(rvgaw.post_samples_phi), col = "red", lty = 2, lwd = 3)
# abline(v = phi, lty = 3)
# legend("topleft", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
#        col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)
# 
# plot(density(mcmcw.post_samples_eta), main = "Posterior of sigma_eta", 
#      col = "blue", lty = 2, lwd = 3)
# lines(density(rvgaw.post_samples_eta), col = "red", lty = 2, lwd = 3)
# abline(v = sigma_eta, lty = 3)
# legend("topright", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
#        col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)

# ####### MCMCE ##########
# mcmce_filepath <- paste0(result_directory, "mcmc_exact_results_n", n, 
#                          "_phi", phi_string, "_", date, ".rds")
# 
# if (rerun_mcmce) {
#   mcmce_results <- run_mcmc_sv(y, #sigma_eta, sigma_eps, 
#                                iters = MCMC_iters, burn_in = burn_in,
#                                prior_mean = prior_mean, prior_var = prior_var,  
#                                state_ini_mean = state_ini_mean, state_ini_var = state_ini_var,
#                                adapt_proposal = T, use_whittle_likelihood = F)
#   
#   if (save_mcmce_results) {
#     saveRDS(mcmce_results, mcmce_filepath)
#   }
# } else {
#   mcmce_results <- readRDS(mcmce_filepath)
# }
# 
# mcmce.post_samples_phi <- as.mcmc(mcmce_results$post_samples$phi[-(1:burn_in)])
# mcmce.post_samples_eta <- as.mcmc(mcmce_results$post_samples$sigma_eta[-(1:burn_in)])
# # mcmce.post_samples_xi <- as.mcmc(mcmce_results$post_samples$sigma_xi[-(1:burn_in)])
# 
# par(mfrow = c(2,1))
# coda::traceplot(mcmce.post_samples_phi, main = "Trace plot for phi")
# coda::traceplot(mcmce.post_samples_eta, main = "Trace plot for sigma_eta")
# # traceplot(mcmce.post_samples_xi, main = "Trace plot for sigma_xi")

### STAN ###
hmc_filepath <- paste0(result_directory, "hmc_results_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

n_post_samples <- 10000
burn_in <- 1000
stan.iters <- n_post_samples + burn_in
transform01 <- 0

if (rerun_hmc) {
  stan_results <- run_hmc_sv(data = y, transform = transform,
                             prior_mean = prior_mean, prior_var = prior_var,
                             iters = stan.iters, burn_in = burn_in)
  
  if (save_hmc_results) {
    saveRDS(stan_results, hmc_filepath)
  }
  
} else {
  stan_results <- readRDS(hmc_filepath)
}


# hmc.fit <- extract(hfit, pars = c("theta_phi", "theta_sigma"),
#                    permuted = F)
# 
# hmc.theta_phi <- hmc.fit[,,1]
# hmc.theta_sigma <- hmc.fit[,,2]

hmc.post_samples_phi <- stan_results$draws[,,1]#tanh(hmc.theta_phi)
hmc.post_samples_eta <- stan_results$draws[,,2]#sqrt(exp(hmc.theta_sigma))


########################################################
##          Stan with the Whittle likelihood          ##
########################################################
hmcw_filepath <- paste0(result_directory, "hmcw_results_n", n, 
                         "_phi", phi_string, prior_type, "_", date, ".rds")

if (rerun_hmcw) {
  
  n_post_samples <- 10000
  burn_in <- 1000
  
  ## Fourier frequencies
  k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
  k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
  freq <- 2 * pi * k_in_likelihood / n

  ## Fourier transform of the observations
  y_tilde <- log(y^2) - mean(log(y^2))

  fourier_transf <- fft(y_tilde)
  periodogram <- 1/n * Mod(fourier_transf)^2
  I <- periodogram[k_in_likelihood + 1]

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
    chains = 1,
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
hmcw.phi <- hmcw_results$draws[,,1]
hmcw.sigma_eta <- hmcw_results$draws[,,2]

###########################

par(mfrow = c(1,2))
plot(density(rvgaw.post_samples_phi), main = "Posterior of phi", 
     col = "red", lty = 2, lwd = 3) #, xlim = c(0.9, 0.999))
# lines(density(mcmce.post_samples_phi), col = "blue", lwd = 3)
# lines(density(mcmcw.post_samples_phi), col = "red", lty = 2, lwd = 3)
lines(density(hmc.post_samples_phi), col = "deepskyblue", lty = 1, lwd = 3)
lines(density(hmcw.phi), col = "royalblue", lty = 2, lwd = 3)
abline(v = phi, lty = 3, lwd = 2)
# legend("topleft", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
      #  col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)

plot(density(rvgaw.post_samples_eta), main = "Posterior of sigma_eta", 
     col = "red", lty = 2, lwd = 3) #, xlim = c(0.05, 0.77))
# lines(density(mcmce.post_samples_eta), col = "blue", lwd = 3)
# lines(density(mcmcw.post_samples_eta), col = "red", lty = 2, lwd = 3)
lines(density(hmc.post_samples_eta), col = "deepskyblue", lty = 1, lwd = 3)
lines(density(hmcw.sigma_eta), col = "royalblue", lty = 2, lwd = 3)
abline(v = sigma_eta, lty = 3, lwd = 2)
# legend("topright", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
      #  col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)


## Trajectories
if (plot_trajectories) {
  mu_phi <- sapply(rvgaw_results$mu, function(x) x[1])
  mu_eta <- sapply(rvgaw_results$mu, function(x) x[2])
  
  par(mfrow = c(1, 2))
  if (transform == "arctanh") {
    plot(tanh(mu_phi), type = "l", main = "Trajectory of phi")
  } else {
    plot(1 / (1 + exp(-mu_phi)), type = "l", main = "Trajectory of phi")
  }
  abline(h = phi, lty = 2)
  
  plot(sqrt(exp(mu_eta)), type = "l", main = "Trajectory of sigma_eta")
  abline(h = sigma_eta, lty = 2)
}

## Estimation of kappa
mean_log_eps2 <- digamma(1/2) + log(2)
log_kappa2 <- mean(log(y^2)) - mean_log_eps2
kappa <- sqrt(exp(log_kappa2))


if (save_plots) {
  plot_file <- paste0("sv_posterior_", n, temper_info, reorder_info, welch_info,
                      "_", transform, "_", date, ".png")
  filepath = paste0("./plots/", plot_file)
  png(filepath, width = 1000, height = 500)
  
  par(mfrow = c(1,2))
  plot(density(rvgaw.post_samples_phi), main = "Posterior of phi", 
      col = "red", lty = 2, lwd = 3) #, xlim = c(0.9, 0.999))
  # lines(density(mcmce.post_samples_phi), col = "blue", lwd = 3)
  # lines(density(mcmcw.post_samples_phi), col = "red", lty = 2, lwd = 3)
  lines(density(hmc.post_samples_phi), col = "deepskyblue", lty = 1, lwd = 3)
  lines(density(hmcw.phi), col = "royalblue", lty = 2, lwd = 3)
  abline(v = phi, lty = 3, lwd = 2)
  # legend("topleft", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
        #  col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)

  plot(density(rvgaw.post_samples_eta), main = "Posterior of sigma_eta", 
      col = "red", lty = 2, lwd = 3) #, xlim = c(0.05, 0.77))
  # lines(density(mcmce.post_samples_eta), col = "blue", lwd = 3)
  # lines(density(mcmcw.post_samples_eta), col = "red", lty = 2, lwd = 3)
  lines(density(hmc.post_samples_eta), col = "deepskyblue", lty = 1, lwd = 3)
  lines(density(hmcw.sigma_eta), col = "royalblue", lty = 2, lwd = 3)
  abline(v = sigma_eta, lty = 3, lwd = 2)
  # legend("topright", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
        #  col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)
  dev.off()
}

## Trace 