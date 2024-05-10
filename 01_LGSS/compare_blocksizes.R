## Compare different block sizes

setwd("~/R-VGA-Whittle/01_LGSS/")
rm(list = ls())

library(mvtnorm)
# library(coda)
# library(Deriv)
library(tensorflow)
# library(cmdstanr)
reticulate::use_condaenv("myenv", required = TRUE)
library(keras)
library(Matrix)
library(ggplot2)
library(grid)
library(gtable)
library(gridExtra)
# library(reshape2)
library(tidyr)

# source("./source/run_rvgaw_lgss_tf.R")
source("./source/run_rvgaw_lgss_block.R")
# source("./source/run_mcmc_lgss_allparams.R")
source("./source/compute_periodogram.R")
source("./source/find_cutoff_freq.R")
# source("./source/compute_kf_likelihood.R")
# source("./source/compute_whittle_likelihood_lgss.R")
# source("./source/update_sigma.R")
# source("./source/run_hmc_lgss.R")

## Result directory
result_directory <- "./results/blocksize_test/"

## Flags
rerun_test <- F
save_rvgaw_results <- F
save_plots <- T

date <- "20230525"
n <- 10000
phi <- 0.9

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

## Read data
print("Reading saved data...")
lgss_data <- readRDS(file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))
y <- lgss_data$y
x <- lgss_data$x
phi <- lgss_data$phi
sigma_eps <- lgss_data$sigma_eps
sigma_eta <- lgss_data$sigma_eta  

## Prior
prior_mean <- c(0, -1, -1)#rep(0, 3)
prior_var <- diag(c(1, 1, 1))

## R-VGAL flags
n_post_samples <- 10000
S <- 1000L
use_tempering <- T
temper_first <- T
reorder <- 0 #"decreasing"
reorder_seed <- 2024
# decreasing <- T
transform <- "arctanh"
power_prop <- 1/10
nsegs <- 20
welch_output <- find_cutoff_freq(y, nsegs = nsegs, power_prop = power_prop)
n_indiv <- welch_output$cutoff_ind #500
blocksizes <- c(10, 50, 100, 300, 500, 1000)

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

## Running R-VGA with different block sizes
rvgaw_post_samples <- list()

if (rerun_test) {

  for (b in 1:length(blocksizes)) {
    blocksize <- blocksizes[b]
    cat("Blocksize =", blocksize, "\n")
    block_info <- paste0("_", "blocksize", blocksize, "_", n_indiv, "indiv")
    rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_", transform, "_n", n,
                            "_phi", phi_string, temper_info, reorder_info, block_info, "_", date, ".rds")
    
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
      rvgaw_post_samples[[b]] <- rvgaw_results$post_samples
  }
  
} else {
  for (b in 1:length(blocksizes)) {
    blocksize <- blocksizes[b]
    cat("Blocksize =", blocksize, "\n")
    block_info <- paste0("_", "blocksize", blocksize, "_", n_indiv, "indiv")
    rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_", transform, "_n", n,
                            "_phi", phi_string, temper_info, reorder_info, block_info, "_", date, ".rds")
    
    rvgaw_results <- readRDS(rvgaw_filepath)
    rvgaw_post_samples[[b]] <- rvgaw_results$post_samples
  } 
}

## Plot posterior densities for different block sizes
param_names <- c("phi", "sigma[eta]", "sigma[epsilon]")
param_dim <- length(param_names)
param_values <- c(phi, sigma_eta, sigma_eps)  

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

df_list <- list()
plots <- list()
# rvgaw_post_samples_df <- matrix(NA, nrow = n_post_samples, ncol = length(blocksizes))

for (p in 1:param_dim) {
  rvgaw_post_samples_list <- lapply(rvgaw_post_samples, function(x) x[[p]])
  rvgaw_post_samples_df <- as.data.frame(do.call(cbind, rvgaw_post_samples_list))

  colnames(rvgaw_post_samples_df) <- sapply(blocksizes, function(x) paste0("blocksize", x))
  df_list[[p]] <- rvgaw_post_samples_df

  rvgaw_post_samples_df_long <- rvgaw_post_samples_df %>% pivot_longer(cols = starts_with("blocksize"),
                    names_to = "blocksize",
                    names_prefix = "blocksize",
                    values_to = "post_samples")
                    
  rvgaw_post_samples_df_long$blocksize <- factor(rvgaw_post_samples_df_long$blocksize,
                              levels = sapply(blocksizes, toString))                    
  # rvgaw_post_samples_df_long$param <- rep(param_names[p], nrow(rvgaw_post_samples_df_long))
  
  # hmcw_df_long <- hmcw_df %>% pivot_longer(cols = everything(),
  #                                       names_to = "param",
  #                                       values_to = "post_samples")
  
  # hmc_df_long <- hmc_df %>% pivot_longer(cols = everything(),
  #                                       names_to = "param",
  #                                       values_to = "post_samples")
  hmcw_post_samples <- data.frame(val = hmcw_df[, p])
  # hmc_post_samples <- data.frame(val = hmc_df[, p])

  ## Plot
  true_vals.df <- data.frame(name = param_names[p], val = param_values[p])

  plot <- ggplot(rvgaw_post_samples_df_long, aes(x = post_samples)) +
    geom_density(aes(col = blocksize), lwd = 1.5) +
    geom_density(data = hmcw_post_samples, aes(x = val), 
                col = "black", linewidth = 1) +
    # geom_density(data = hmc_post_samples, aes(x = val), col = "deepskyblue", lwd = 1) +
    geom_vline(data = true_vals.df, aes(xintercept=val),
               color="black", linetype="dashed", linewidth=1) +
    # geom_density(aes(x = hmcw_df[, p]), col = "goldenrod") +
    # geom_density(aes(x = hmc_df[, p]), col = "cyan") +
    labs(x = str2expression(param_names[p])) +
    theme_bw() +
    theme(text = element_text(size = 24)) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 4))
  
  plots[[p]] <- plot  
}

grid.arrange(grobs = plots, nrow = 1, ncol = 3)

if (save_plots) {
  plot_file <- paste0("compare_blocksizes_lgss_", n, temper_info, reorder_info, 
                      # "_", n_indiv, "indiv",
                      "_power", 1/power_prop,
                      "_", transform, "_", date, ".png")
  filepath = paste0("./plots/blocksize_test/", plot_file)
  png(filepath, width = 2000, height = 500)
  grid.arrange(grobs = plots, nrow = 1, ncol = 3)
  dev.off()
} 

# ## Plot periodogram
pdg_plot <- welch_output$pdg_plot 

# pgram_output <- compute_periodogram(y)
# freq <- pgram_output$freq
# psd <- pgram_output$periodogram

# welch_output <- compute_welch_psd(y, nperseg = length(y)/nsegs, p_overlap = 0.5)
# freq_welch <- welch_output$freq
# pdg_welch <- welch_output$pdg

# cutoff_freq_og <- welch_output$cutoff_freq_og
# #(v = cutoff_freq_og, lty = 2, lwd = 2, col = "red")

# pdg_df <- data.frame(freq = freq, pdg = pdg)
# pdg_welch_df <- data.frame(freq = freq_welch, pdg = pdg_welch)

# pg_plot <- pdg_df %>% ggplot(aes(x = freq, y = pdg)) +
#           geom_line() +
#           geom_line(data = pdg_welch_df, aes(x = freq, y = pdg), 
#                       color = "salmon", linewidth = 1.5) +
#           geom_vline(xintercept = cutoff_freq_og, 
#                       linetype = 2, color = "red", linewidth = 1) +
#           labs(x = "Frequency", y = "Periodogram") +
#           xlim(c(0, 1)) +
#           theme_bw() +
#           theme(text = element_text(size = 24))

# print(pg_plot)

if (save_plots) {
  periodogram_plot <- paste0("periodogram_lgss_power", 1/power_prop, "_", date, ".png")
  filepath = paste0("./plots/blocksize_test/", periodogram_plot)
  png(filepath, width = 800, height = 400)
  print(pdg_plot)
  dev.off()
}

## Plot all cutoff freqs in one plot
welch_power2 <- find_cutoff_freq(y, nsegs = nsegs, power_prop = 1/2)
welch_power5 <- find_cutoff_freq(y, nsegs = nsegs, power_prop = 1/5)
welch_power10 <- find_cutoff_freq(y, nsegs = nsegs, power_prop = 1/10)

power2_cutoff <- welch_power2$cutoff_freq
power5_cutoff <- welch_power5$cutoff_freq
power10_cutoff <- welch_power10$cutoff_freq

welch_pdg_df <- data.frame(freq = welch_power2$freq_welch, pdg = welch_power2$pdg_welch)
# welch_power5_df <- data.frame(freq = welch_power5$freq_welch, pdg = welch_power5$pdg_welch)
# welch_power10_df <- data.frame(freq = welch_power10$freq_welch, pdg = welch_power10$pdg_welch)

pdg_og <- compute_periodogram(y)
pdg_df <- data.frame(freq = pdg_og$freq, pdg = pdg_og$periodogram)

pdg_plot <- pdg_df %>% ggplot(aes(x = freq, y = pdg)) +
                        geom_line() +
                        geom_line(data = welch_pdg_df, aes(x = freq, y = pdg), 
                                    color = "salmon", linewidth = 1.5) +
                        geom_vline(xintercept = power2_cutoff, 
                                    linetype = 2, color = "red", linewidth = 1) +
                        # geom_line(data = welch_power5_df, aes(x = freq, y = pdg), 
                        #             color = "salmon", linewidth = 1.5) +
                        geom_vline(xintercept = power5_cutoff, 
                                    linetype = 2, color = "blue", linewidth = 1) +
                        geom_vline(xintercept = power10_cutoff, 
                                    linetype = 2, color = "green", linewidth = 1) +
                        labs(x = "Frequency (rad/s)", y = "Power") +
                        xlim(c(0, 1)) +
                        theme_bw() +
                        theme(text = element_text(size = 24))

print(pdg_plot)


