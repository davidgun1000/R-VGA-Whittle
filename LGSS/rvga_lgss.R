setwd("~/R-VGA-whittle/LGSS/")
rm(list = ls())

# library("mvtnorm")
source("./source/run_rvgaw_lgss.R")
# source("./source/compute_kf_likelihood.R")
# source("./source/compute_whittle_likelihood_lgss.R")
# source("./source/update_sigma.R")

library(coda)

## Flags
date <- "20230525"
regenerate_data <- F
save_data <- F
use_tempering <- T

## True parameters
sigma_eps <- 0.1 # measurement error var
sigma_eta <- 0.5 # process error var
phi <- 0.7

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

# Generate true process x_1:T
n <- 1000
# times <- seq(0, 1, length.out = iters)

if (regenerate_data) {
  x <- c()
  x[1] <- 1
  set.seed(2023)
  for (t in 2:n) {
    x[t] <- phi * x[t-1] + rnorm(1, 0, sigma_eta)
  }
  
  # Generate observations y_1:T
  y <- x + rnorm(n, 0, sigma_eps)
  
  ## Plot true process and observations
  # par(mfrow = c(1, 1))
  # plot(x, type = "l", main = "True process")
  # points(y, col = "cyan")
  
  lgss_data <- list(x = x, y = y, phi = phi, sigma_eps = sigma_eps, sigma_eta = sigma_eta)
  if (save_data) {
    saveRDS(lgss_data, file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  }
} else {
  lgss_data <- readRDS(file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  y <- lgss_data$y
  x <- lgss_data$x
  phi <- lgss_data$phi
  sigma_eps <- lgss_data$sigma_eps
  sigma_eta <- lgss_data$sigma_eta
}


## R-VGA with Whittle likelihood on LGSS

if (use_tempering) {
  n_temper <- 0.5 * n
  temper_schedule <- rep(1/10, 10)
}

## Prior
prior_mean <- 0
prior_var <- 1

rvgaw_results <- run_rvgaw_lgss(y = y, sigma_eta = sigma_eta, sigma_eps = sigma_eps, 
                                prior_mean = prior_mean, prior_var = prior_var, 
                                S = 500, use_tempering = use_tempering, 
                                n_temper = n_temper,
                                temper_schedule = temper_schedule)

# if (save_rvgaw_results) {
#   saveRDS(rvgaw_results, rvgaw_filepath)
# }

rvgaw.post_samples <- rvgaw_results$post_samples
par(mfrow = c(1,1))
plot(density(rvgaw.post_samples), main = "Posterior of phi")
abline(v = phi, lty = 2)