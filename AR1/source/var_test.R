## Variance test for R-VGA Whittle

setwd("~/R-VGA-whittle/AR1")

## R-VGA with Whittle likelihood (R-VGAW)?
rm(list = ls())

# library(stats)
# library(LSTS)
library(Matrix)
# library(coda)
library(ggplot2)
library(tidyr)
source("calculate_likelihood.R")
source("run_rvgaw_ar1.R")
source("run_rvgae_ar1.R")
# source("run_mcmc_ar1.R")
source("run_vb_ar1.R")

result_directory <- "./var_test/"

## Flags
date <- "20230417" # 20240410 has phi = 0.9, 20230417 has phi = 0.7
regenerate_data <- T
save_data <- F
use_tempering <- T
reorder_freq <- F
use_matlab_deriv <- T
save_results <- T

## Model parameters 
phi <- 0.9
sigma_e <- 0.5
n <- 10000 # time series length

if (use_tempering) {
  n_temper <- 0.1 * n #floor(n/2) #10
  temper_schedule <- rep(1/10, 10)
  temper_info <- paste0("_temper", n_temper)
} else {
  n_temper <- 0
  temper_schedule <- NULL
  temper_info <- ""
}

if (reorder_freq) {
  reorder_info <- "_reordered"
} else {
  reorder_info <- ""
}

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

if (regenerate_data) {
  ## Generate AR(1) series
  x0 <- 1
  x <- c()
  x[1] <- x0
  # set.seed(2023)
  for (t in 2:n) {
    x[t] <- phi * x[t-1] + rnorm(1, 0, sigma_e) 
  }
  rvgaw_data <- list(x = x, phi = phi, sigma_e = sigma_e)
  
  if (save_data) {
    saveRDS(rvgaw_data, file = paste0("./data/ar1_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  }
  
} else {
  rvgaw_data <- readRDS(file = paste0("./data/ar1_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  x <- rvgaw_data$x
  phi <- rvgaw_data$phi
  sigma_e <- rvgaw_data$sigma_e
}

plot(1:n, x, type = "l", main = "Time series")

####################################################
##         R-VGA with Whittle likelihood          ##
####################################################

# rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_n", n, 
#                          "_phi", phi_string, reorder_info, "_", date, ".rds")

runs <- 5
S <- 500
mu_0 <- 0
P_0 <- 1
r <- 1

rvgae_results <- list()
rvgaw_results <- list()
rvgaw_post_samples <- list()
rvgae_post_samples <- list()

while (r <= runs) {
  ## Run R-VGA here
  rvgaw_results[[r]] <- run_rvgaw_ar1(series = x, sigma_e = sigma_e,
                                 prior_mean = mu_0, prior_var = P_0, S = S,
                                 use_matlab_deriv = use_matlab_deriv)
  rvgaw_post_samples[[r]] <- rvgaw_results[[r]]$post_samples

  try({
    rvgae_results[[r]] <- run_rvgae_ar1(series = x, sigma_e = sigma_e, 
                                   prior_mean = mu_0, prior_var = P_0, S = S)
    rvgae_post_samples[[r]] <- rvgae_results[[r]]$post_samples
    r <- r+1
  })
  
}

## Batch VB posterior for comparison
print("Starting batch VB with exact likelihood...")
vbe_results <- run_vb_ar1(series = x, mu_0 = mu_0, sigma_0 = P_0, 
                          sigma_e = sigma_e, 
                          VB_iters = 2000, use_whittle_likelihood = F)
vbe_post_samples <- vbe_results$post_samples

## Plot results
# run_index <- rep(1:runs, each = length(rvgae_post_samples[[1]]))
# all_rvgae_post_samples <- unlist(rvgae_post_samples)
# 
# post_samples_df <- data.frame(run = run_index,
#                               rvgae_post_samples = all_rvgae_post_samples,
#                               vbe_post_samples = vbe_post_samples)
# post_samples_df$run <- as.factor(post_samples_df$run)
# 
# 
# plot <- ggplot(post_samples_df, aes(x = rvgae_post_samples, group = run, col = "rvga-exact")) + #geom_line(aes(colour = series))
#   geom_density() +
#   geom_density(data = post_samples_df, aes(x = vbe_post_samples, col = "vb-exact")) +
#   # geom_density(data = post_samples_df, aes(x = rvgaw_post_samples, group = run, col = "rvga-whittle")) +
#   geom_vline(aes(xintercept=phi), lty = 2) +
#   theme_bw()
# print(plot)


## Plot the densities
run_index <- rep(1:runs, each = length(rvgaw_post_samples[[1]]))
all_rvgaw_post_samples <- unlist(rvgaw_post_samples)
all_rvgae_post_samples <- unlist(rvgae_post_samples)

post_samples_df <- data.frame(run = run_index,
                              rvgaw_post_samples = all_rvgaw_post_samples,
                              rvgae_post_samples = all_rvgae_post_samples,
                              vbe_post_samples = vbe_post_samples)
post_samples_df$run <- as.factor(post_samples_df$run)

plot <- ggplot(post_samples_df,
               aes(x = rvgaw_post_samples, group = run, col = "rvga-whittle")) + #geom_line(aes(colour = series))
        geom_density() +
        geom_density(data = post_samples_df, aes(x = vbe_post_samples, col = "vb-exact")) +
        geom_density(data = post_samples_df,
                     aes(x = rvgae_post_samples, group = run, col = "rvga-exact")) +
        geom_vline(aes(xintercept=phi), lty = 2) +
        theme_bw() +
        xlab("phi")
print(plot)

## Plot trajectories
trajectories <- matrix(NA, nrow = length(rvgae_results[[1]]$mu), ncol = runs)
for (r in 1:runs) {
  trajectories[, r] <- tanh(unlist(rvgae_results[[r]]$mu))
  # trajectories[, r] <- unlist(rvgae_results[[r]]$mu)
}
plot_range <- 1:1000
matplot(trajectories[plot_range, ], type = "l")
abline(h = phi, lty = 2)

## Save plot and results
if (save_results) {
  var_test_results <- list(rvgae_results = rvgae_results,
                           rvgaw_results = rvgaw_results)
  results_file <- paste0("var_test_n", n, "_phi", phi_string, 
                         temper_info, reorder_info, "_", date, ".rds")
  saveRDS(var_test_results, file = paste0(result_directory, results_file))
  
  png(paste0(result_directory, "var_test_n", n, "_phi", phi_string, 
                          temper_info, reorder_info, "_", date, ".png"))
  print(plot)
  dev.off()
}

