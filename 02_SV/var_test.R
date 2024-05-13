## Variance test for R-VGA-Whittle on a univariate stochastic volatility model
setwd("~/R-VGA-Whittle/02_SV/")

rm(list = ls())

library(mvtnorm)
# library(coda)
# library(Deriv)
# library(cmdstanr)
library(tensorflow)
reticulate::use_condaenv("myenv", required = TRUE)
library(keras)
library(stats)
library(bspec)
library(ggplot2)
# library(grid)
library(tidyr)
library(gridExtra)
# library(gtable)

# source("./source/compute_whittle_likelihood_sv.R")
# source("./source/run_rvgaw_sv_tf.R")
source("./source/run_rvgaw_sv_block.R")
# source("./source/run_mcmc_sv.R")
# source("./source/run_hmc_sv.R")
source("./source/compute_periodogram.R")
source("./source/find_cutoff_freq.R")
# source("./source/run_corr_pmmh_sv.R")
# source("./source/particleFilter.R")

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
rerun_rvgaw <- F
save_results <- F
use_tempering <- T
temper_first <- T
reorder <- 0 #"decreasing" # or decreasing # or a number
reorder_seed <- 2024

## Read data
date <- "20240214"
phi <- 0.99
sigma_eta <- 0.1 #sqrt(0.1)
sigma_eps <- 1
kappa <- 2
n <- 10000

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

print("Reading saved data...")
sv_data <- readRDS(file = paste0("./data/sv_data_n", n, "_phi", phi_string, "_", date, ".rds"))

y <- sv_data$y
x <- sv_data$x
phi <- sv_data$phi
sigma_eta <- sv_data$sigma_eta
sigma_eps <- sv_data$sigma_eps


## Prior
prior_mean <- c(2, -3) #rep(0,2)
prior_var <- diag(c(0.5, 0.5)) #diag(1, 2)

########################################
##                R-VGA               ##
########################################

runs <- 10
S <- 100L
n_post_samples <- 10000
blocksize <- 500
power_prop <- 1/2
n_indiv <- find_cutoff_freq(y, nsegs = 25, power_prop = power_prop)$cutoff_ind #500
# n_indiv <- 100
transform <- "arctanh"

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

result_directory <- "./var_test/results/"

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_n", n, "_S", S,
                         temper_info, reorder_info, block_info, "_", date, ".rds")

rvgaw_results <- list()

if (rerun_rvgaw) {

    for (r in 1:runs) {
        rvgaw_results[[r]] <- run_rvgaw_sv(y = y, #sigma_eta = sigma_eta, sigma_eps = sigma_eps, 
                                prior_mean = prior_mean, prior_var = prior_var, 
                                deriv = "tf", 
                                n_post_samples = n_post_samples,
                                S = S, use_tempering = use_tempering, 
                                temper_first = temper_first,
                                reorder = reorder,
                                n_temper = n_temper,
                                temper_schedule = temper_schedule, 
                                transform = transform,
                                # nblocks = nblocks,
                                blocksize = blocksize,
                                n_indiv = n_indiv)

    }
    
  if (save_results) {
    saveRDS(rvgaw_results, rvgaw_filepath)
  }
  
} else {
  rvgaw_results <- readRDS(rvgaw_filepath)
}

## Extract posterior samples
param_names <- c("phi", "sigma[eta]")
true_df <- data.frame(param = param_names, value = c(phi, sigma_eta))

## HMCW and HMC results
hmcw_filepath <- paste0("~/R-VGA-Whittle/02_SV/results/arctanh/hmcw_results_n", n, 
                       "_phi", phi_string, "_", date, ".rds")
hmc_filepath <- paste0("~/R-VGA-Whittle/02_SV/results/arctanh/hmc_results_n", n, 
                       "_phi", phi_string, "_", date, ".rds")

hmcw_results <- readRDS(hmcw_filepath)
hmc_results <- readRDS(hmc_filepath)

hmcw_draws <- apply(hmcw_results$draws, 3, c)
hmc_draws <- apply(hmc_results$draws, 3, c)

hmcw_df <- as.data.frame(hmcw_draws)
hmc_df <- as.data.frame(hmc_draws)
names(hmcw_df) <- param_names
names(hmc_df) <- param_names

long_hmcw_df <- hmcw_df %>% pivot_longer(cols = everything(), names_to = "param", values_to = "value")
long_hmc_df <- hmc_df %>% pivot_longer(cols = everything(), names_to = "param", values_to = "value")

## R-VGA-Whittle results

df_list <- list()
for (r in 1:runs) {

  rvgaw.post_samples <- do.call(rbind, rvgaw_results[[r]]$post_samples)
  rvgaw.df <- as.data.frame(t(rvgaw.post_samples))
  names(rvgaw.df) <- param_names
  rvgaw.df$run <- rep(r, nrow(rvgaw.df))

  df_list[[r]] <- rvgaw.df
}

all_df <- do.call(rbind, df_list)
long_df <- all_df %>% pivot_longer(cols = !run, names_to = "param", values_to = "value")

plot <- long_df %>% ggplot(aes(x = value, col = as.factor(run))) + geom_density(linewidth = 1.5) + 
  facet_wrap(~param, scales = "free", labeller = label_parsed) +
  geom_density(data = long_hmcw_df, aes(x = value), col = "black", linewidth = 1.5, linetype = 2) +
  geom_density(data = long_hmc_df, aes(x = value), col = "black", linewidth = 1.5, linetype = 3) +
  geom_vline(data = true_df, aes(xintercept = value), linetype = "dashed", linewidth = 1.5) +
  theme_bw() +
  theme(text = element_text(size = 24), legend.position="none")      
print(plot)

if (save_results) {
  plot_name <- paste0("var_test_sv_S", S, "_power", 1/power_prop, block_info, "_", date, ".png")
  png(paste0("./var_test/plots/", plot_name), width = 1000, height = 500)
  print(plot)
  dev.off()
}

