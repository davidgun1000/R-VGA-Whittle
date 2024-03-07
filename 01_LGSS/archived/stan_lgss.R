setwd("~/R-VGA-Whittle/LGSS/")
rm(list = ls())

# library("mvtnorm")
source("./source/run_rvgaw_lgss.R")
source("./source/run_mcmc_lgss.R")
source("./source/run_hmc_lgss.R")
# source("./source/compute_kf_likelihood.R")
# source("./source/compute_whittle_likelihood_lgss.R")
# source("./source/update_sigma.R")

library(coda)
library(cmdstanr)
result_directory <- "./results/"

## Flags
date <- "20230525"
regenerate_data <- F
save_data <- F

rerun_hmc <- T
save_hmc_results <- F

## True parameters
sigma_eps <- 0.2 # measurement error var
sigma_eta <- 0.7 # process error var
phi <- 0.8

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

n <- 1000

if (regenerate_data) {
  print("Generating data...")
  
  # Generate true process x_0:T
  x <- c()
  x0 <- rnorm(1, 0, sqrt(sigma_eta^2 / (1-phi^2)))
  x[1] <- x0
  set.seed(2023)
  for (t in 2:(n+1)) {
    x[t] <- phi * x[t-1] + rnorm(1, 0, sigma_eta)
  }
  
  # Generate observations y_1:T
  y <- x[2:(n+1)] + rnorm(n, 0, sigma_eps)
  
  ## Plot true process and observations
  # par(mfrow = c(1, 1))
  plot(x, type = "l", main = "True process")
  # points(y, col = "cyan")
  
  lgss_data <- list(x = x, y = y, phi = phi, sigma_eps = sigma_eps, sigma_eta = sigma_eta)
  if (save_data) {
    saveRDS(lgss_data, file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  }
} else {
  print("Reading saved data...")
  lgss_data <- readRDS(file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  y <- lgss_data$y
  x <- lgss_data$x
  phi <- lgss_data$phi
  sigma_eps <- lgss_data$sigma_eps
  sigma_eta <- lgss_data$sigma_eta
}

## MCMC settings
burn_in <- 5000
n_post_samples <- 10000
# MCMC_iters <- n_post_samples + burn_in # Number of MCMC iterations

# ## Prior
# prior_mean <- rep(0, 3)
# prior_var <- diag(1, 3)
# 
# ## Initial state mean and variance for the KF
# state_ini_mean <- 0
# state_ini_var <- 1


#########################
###        STAN       ###
#########################

hmc_filepath <- paste0(result_directory, "hmc_results_n", n, 
                       "_phi", phi_string, "_", date, ".rds")

# n_post_samples <- 10000
# burn_in <- 1000
stan.iters <- n_post_samples + burn_in

if (rerun_hmc) {
  hmc_results <- run_hmc_lgss(data = y, iters = stan.iters, burn_in = burn_in)
  
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

hmc.phi <- hmc_results$draws[,,1]#tanh(hmc.theta_phi)
hmc.sigma_eta <- hmc_results$draws[,,2]#sqrt(exp(hmc.theta_sigma))
hmc.sigma_eps <- hmc_results$draws[,,3]#sqrt(exp(hmc.theta_sigma))


########################################################
##          Stan with the Whittle likelihood          ##
########################################################

## Fourier frequencies
k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
freq <- 2 * pi * k_in_likelihood / n

## Fourier transform of the observations
fourier_transf <- fft(y)
periodogram <- 1/n * Mod(fourier_transf)^2
I <- periodogram[k_in_likelihood + 1]

whittle_stan_file <- "./source/stan_lgss_whittle.stan"

whittle_lgss_model <- cmdstan_model(
  whittle_stan_file,
  cpp_options = list(stan_threads = TRUE)
)

# log_kappa2_est <- mean(log(y^2)) - (digamma(1/2) + log(2))

whittle_lgss_data <- list(nfreq = length(freq), freqs = freq, periodogram = I)

# hfit <- stan(model_code = lgss_code, 
#              model_name="lgss", data = lgss_data, 
#              iter = iters, warmup = burn_in, chains=1)

fit_stan_lgss_whittle <- whittle_lgss_model$sample(
  whittle_lgss_data,
  chains = 1,
  threads = parallel::detectCores(),
  refresh = 5,
  iter_warmup = burn_in,
  iter_sampling = n_post_samples
)

hmcw_results <- list(draws = fit_stan_lgss_whittle$draws(variables = c("phi", "sigma_eta", "sigma_eps")),
                     time = fit_stan_lgss_whittle$time,
                     summary = fit_stan_lgss_whittle$cmdstan_summary)
fit_stan_lgss_whittle$cmdstan_summary()

fit_stan_lgss_whittle$diagnostic_summary()

# hmcw.fit <- extract(hfit, pars = c("theta_phi", "theta_sigma"),
#                    permuted = F)
# 
hmcw.phi <- hmcw_results$draws[,,1]
hmcw.sigma_eta <- hmcw_results$draws[,,2]
hmcw.sigma_eps <- hmcw_results$draws[,,3]

# hmcw.phi <- tanh(hmcw.theta_phi)
# hmcw.sigma_eta <- sqrt(exp(hmcw.theta_sigma))

par(mfrow = c(1,3))
plot(density(hmc.phi), main = "Posterior of phi")
lines(density(hmcw.phi), col = "red")
abline(v = phi, lty = 2)

plot(density(hmc.sigma_eta), main = "Posterior of sigma_eta")
lines(density(hmcw.sigma_eta), col = "red")
abline(v = sigma_eta, lty = 2)

plot(density(hmc.sigma_eps), main = "Posterior of sigma_eps")
lines(density(hmcw.sigma_eps), col = "red")
abline(v = sigma_eps, lty = 2)

# traceplot(hwhittle.fit, c("theta_phi", "theta_sigma"),
#           ncol=1,nrow=2,inc_warmup=F)

