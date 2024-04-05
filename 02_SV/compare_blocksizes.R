## Compare different block sizes

setwd("~/R-VGA-Whittle/02_SV/")
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
source("./source/run_rvgaw_sv_block.R")
# source("./source/run_mcmc_lgss_allparams.R")
source("./source/compute_periodogram.R")
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

date <- 20240214
n <- 10000
phi <- 0.99

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

## R-VGAL flags
n_post_samples <- 20000
S <- 1000L
use_tempering <- T
temper_first <- T
reorder <- 0 #"decreasing"
reorder_seed <- 2024
# decreasing <- T
transform <- "arctanh"
n_indiv <- 100
blocksizes <- c(10, 50, 100, 300, 500, 1000)
# blocksizes <- 1000

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

## Read data
print("Reading saved data...")
sv_data <- readRDS(file = paste0("./data/sv_data_n", n, "_phi", phi_string, "_", date, ".rds"))
y <- sv_data$y
x <- sv_data$x
phi <- sv_data$phi
sigma_eta <- sv_data$sigma_eta
sigma_eps <- sv_data$sigma_eps

## Prior
prior_mean <- c(2, -3) #rep(0,2)
# prior_mean <- c(0, -3) #rep(0,2)
prior_var <- diag(c(0.5, 0.5)) #diag(1, 2)

## Running R-VGA with different block sizes
rvgaw_post_samples <- list()

if (rerun_test) {

  for (b in 1:length(blocksizes)) {
    blocksize <- blocksizes[b]
    cat("Blocksize =", blocksize, "\n")
    block_info <- paste0("_", "blocksize", blocksize, "_", n_indiv, "indiv")
    rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_n", n, 
                         "_phi", phi_string, temper_info, reorder_info, block_info,
                         "_", date, ".rds")

    rvgaw_results <- run_rvgaw_sv(y = y, #sigma_eta = sigma_eta, sigma_eps = sigma_eps, 
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
    rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_n", n, 
                         "_phi", phi_string, temper_info, reorder_info, block_info,
                         "_", date, ".rds")
    rvgaw_results <- readRDS(rvgaw_filepath)
    rvgaw_post_samples[[b]] <- rvgaw_results$post_samples
  } 
}

## Plot posterior densities for different block sizes
param_names <- c("phi", "sigma[eta]")
param_dim <- length(param_names)

param_values <- c(phi, sigma_eta)  

df_list <- list()
plots <- list()
# rvgaw_post_samples_df <- matrix(NA, nrow = n_post_samples, ncol = length(blocksizes))

p <- 1
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
  
  ## Plot
  true_vals.df <- data.frame(name = param_names[p], val = param_values[p])

  plot <- ggplot(rvgaw_post_samples_df_long, aes(x = post_samples)) +
    geom_density(aes(col = blocksize), lwd = 1.5) +
    # geom_density(data = hmcw.df, col = "goldenrod", lwd = 1) +
    # geom_density(data = hmc.df, col = "deepskyblue", lwd = 1) +
    geom_vline(data = true_vals.df, aes(xintercept=val),
               color="black", linetype="dashed", linewidth=1) +
    labs(x = str2expression(param_names[p])) +
    theme_bw() +
    theme(text = element_text(size = 24)) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 4))
  
  plots[[p]] <- plot  
}

grid.arrange(grobs = plots, nrow = 1, ncol = param_dim)

if (save_plots) {
  plot_file <- paste0("compare_blocksizes_sv_sim_", n, temper_info, reorder_info,
                      "_", n_indiv, "indiv",
                      "_", transform, "_", date, ".png")
  filepath = paste0("./plots/blocksize_test/", plot_file)
  png(filepath, width = 1200, height = 500)
  grid.arrange(grobs = plots, nrow = 1, ncol = param_dim)
  dev.off()
} 

## Plot periodogram 
pgram_output <- compute_periodogram(y)
freq <- pgram_output$freq
I <- pgram_output$periodogram

periodogram_plot <- paste0("periodogram_sv_sim_", date, ".png")

if (save_plots) {
  filepath = paste0("./plots/blocksize_test/", periodogram_plot)
  png(filepath, width = 800, height = 400)
  plot(freq, I, type = "l", xlab = "Frequency", ylab = "Periodogram")
  abline(v = freq[n_indiv], lwd = 2, lty = 2, col = "red")
  dev.off()
}
