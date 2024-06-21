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
library(ggplot2)
library(tidyr)
library(gridExtra)
# library(grid)
# library(gtable)
# library(gridExtra)

# source("./source/run_rvgaw_lgss_tf.R")
source("./source/run_rvgaw_lgss_block.R")
# source("./source/run_mcmc_lgss_allparams.R")
source("./source/compute_periodogram.R")
# source("./source/compute_kf_likelihood.R")
source("./source/compute_whittle_likelihood_lgss.R")
source("./source/find_cutoff_freq.R")
# source("./source/update_sigma.R")
# source("./source/run_hmc_lgss.R")

rerun_rvgaw <- F
save_results <- F
use_tempering <- T
temper_first <- T
reorder <- 0 #"decreasing"
reorder_seed <- 2024
# decreasing <- T
transform <- "arctanh"

print("Reading saved data...")
date <- "20230525"
n <- 10000
phi <- 0.9
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

lgss_data <- readRDS(file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))

y <- lgss_data$y
x <- lgss_data$x
phi <- lgss_data$phi
sigma_eps <- lgss_data$sigma_eps
sigma_eta <- lgss_data$sigma_eta

## Prior
prior_mean <- c(0, -1, -1)#rep(0, 3)
prior_var <- diag(c(1, 1, 1))

################################################################################
##                      R-VGA with Whittle likelihood                         ##
################################################################################

n_post_samples <- 10000
runs <- 10
# mc_samples <- c(100L, 500L, 1000L, 5000L)
S <- 100L

# nblocks <- 100
power_prop <- 1/2
n_indiv <- find_cutoff_freq(y, nsegs = 25, power_prop = power_prop)$cutoff_ind #500
# n_indiv <- 1000 #807
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

result_directory <- "./var_test/results/"

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_n", n, "_S", S,
                         temper_info, reorder_info, block_info, "_", date, ".rds")

rvgaw_results <- list()

if (rerun_rvgaw) {
  r <- 1
    # for (r in 1:runs) {
  while(r <= runs) {
    # S <- mc_samples[s]
    try ({
      rvgaw_results[[r]] <- run_rvgaw_lgss(y = y, #sigma_eta = sigma_eta, sigma_eps = sigma_eps, 
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
                              n_indiv = n_indiv)
      r <- r+1  
    })
  }
  
    if (save_results) {
        saveRDS(rvgaw_results, rvgaw_filepath)
    }

} else {
  rvgaw_results <- readRDS(rvgaw_filepath)
}

## Extract posterior samples
param_names <- c("phi", "sigma[eta]", "sigma[epsilon]")
true_df <- data.frame(param = param_names, value = c(phi, sigma_eta, sigma_eps))

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

## HMCW and HMC results
hmcw_filepath <- paste0("~/R-VGA-Whittle/01_LGSS/results/hmcw_results_n", n, 
                       "_phi", phi_string, "_", date, ".rds")
hmc_filepath <- paste0("~/R-VGA-Whittle/01_LGSS/results/hmc_results_n", n, 
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

plot <- long_df %>% ggplot(aes(x = value, col = as.factor(run))) + geom_density(linewidth = 1.5) + 
  facet_wrap(~param, scales = "free", labeller = label_parsed) +
  geom_density(data = long_hmcw_df, aes(x = value), col = "black", linewidth = 1.5, linetype = 2) +
  geom_density(data = long_hmc_df, aes(x = value), col = "black", linewidth = 1.5, linetype = 3) +
  geom_vline(data = true_df, aes(xintercept = value), linetype = "dashed", linewidth = 1.5) +
  theme_bw() +
  theme(text = element_text(size = 24), legend.position="none")      
print(plot)

if (save_results) {
  plot_name <- paste0("var_test_lgss_S", S, "_power", 1/power_prop, block_info, "_", date, ".png")
  png(paste0("./var_test/plots/", plot_name), width = 1500, height = 500)
  print(plot)
  dev.off()
}


