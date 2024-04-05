## Compare different block sizes

setwd("~/R-VGA-Whittle/05_Multi_SV_real/")
rm(list = ls())

library(mvtnorm)
# library(coda)
# library(Deriv)
library(tensorflow)
# library(cmdstanr)
reticulate::use_condaenv("myenv", required = TRUE)
library(keras)
library(Matrix)
library(astsa)
# library(stcos)
library(ggplot2)
library(grid)
library(gtable)
library(gridExtra)
# library(reshape2)
library(tidyr)

# List physical devices
gpus <- tf$config$experimental$list_physical_devices('GPU')

if (length(gpus) > 0) {
  tryCatch({
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    tf$config$experimental$set_virtual_device_configuration(
      gpus[[1]],
      list(tf$config$experimental$VirtualDeviceConfiguration(memory_limit=5*4096))
    )
    
    logical_gpus <- tf$config$experimental$list_logical_devices('GPU')
    
    print(paste0(length(gpus), " Physical GPUs,", length(logical_gpus), " Logical GPUs"))
  }, error = function(e) {
    # Virtual devices must be set before GPUs have been initialized
    print(e)
  })
}

# source("./source/run_rvgaw_lgss_tf.R")
source("./source/run_rvgaw_multi_sv_block.R")
# source("./source/run_mcmc_lgss_allparams.R")
source("./source/compute_periodogram.R")
source("./source/construct_prior2.R")
source("./source/compute_grad_hessian_block.R")
source("./source/construct_Sigma.R")

# source("./source/compute_kf_likelihood.R")
# source("./source/compute_whittle_likelihood_lgss.R")
# source("./source/update_sigma.R")
# source("./source/run_hmc_lgss.R")

## Flags
rerun_test <- F
save_rvgaw_results <- F
save_plots <- T

date <- "20240115"
Tfin <- 10000
d <- 2 ## time series dimension
dataset <- "5" # "hmc_est"

# phi <- 0.99

## Result directory
result_directory <- paste0("./results/forex/", d, "d/arctanh/blocksize_test/")

## R-VGAL flags
n_post_samples <- 20000
S <- 1000L
use_tempering <- T
temper_first <- T
use_cholesky <- T
reorder <- 0 # "decreasing"
reorder_seed <- 2024
# decreasing <- T
transform <- "arctanh"
prior_type <- "prior1"
n_indiv <- 100
blocksizes <- c(10, 50, 100, 300, 500, 1000) 
# blocksizes <- 1000

if (use_tempering) {
  n_temper <- 5
  K <- 100
  temper_schedule <- rep(1 / K, K)
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
nstocks <- 2

## Exchange rate data
load("./data/exrates.RData")

data <- dat[, c("AUD", "NZD", "USD")]
nrows <- nrow(data)

# Compute log returns
data$AUD_returns <- c(0, log(data$AUD[2:nrows] / data$AUD[1:(nrows-1)])*100)
data$NZD_returns <- c(0, log(data$NZD[2:nrows] / data$NZD[1:(nrows-1)])*100)
data$USD_returns <- c(0, log(data$USD[2:nrows] / data$USD[1:(nrows-1)])*100)

exrates <- data[-1, c("AUD_returns", "NZD_returns", "USD_returns")] # get rid of 1st row
# Y <- exrates[, 1:nstocks]
Y <- exrates[, c("AUD_returns", "USD_returns")]
Y_demeaned <- Y - colMeans(Y)

## Prior
prior <- construct_prior(data = Y_demeaned, prior_type = prior_type, use_cholesky = use_cholesky)
prior_mean <- prior$prior_mean
prior_var <- prior$prior_var

## Running R-VGA with different block sizes
rvgaw_post_samples <- list()

if (rerun_test) {
  for (b in 1:length(blocksizes)) {
    blocksize <- blocksizes[b]
    cat("Blocksize =", blocksize, "\n")
    block_info <- paste0("_", "blocksize", blocksize, "_", n_indiv, "indiv")
    rvgaw_filepath <- paste0(result_directory, "rvga_whittle_forex",  
                         temper_info, reorder_info, block_info, "_", date, ".rds")

    rvgaw_results <- run_rvgaw_multi_sv(data = Y_demeaned, prior_mean = prior_mean, 
                                      prior_var = prior_var, S = S,
                                      use_cholesky = use_cholesky,
                                      use_tempering = use_tempering, 
                                      temper_schedule = temper_schedule, 
                                      temper_first = temper_first,
                                      reorder = reorder, 
                                      # decreasing = decreasing,
                                      reorder_seed = reorder_seed,
                                      # nblocks = nblocks,
                                      blocksize = blocksize,
                                      n_indiv = n_indiv)
    # use_median = use_median)

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
    rvgaw_filepath <- paste0(result_directory, "rvga_whittle_forex",  
                      temper_info, reorder_info, block_info, "_", date, ".rds")

    rvgaw_results <- readRDS(rvgaw_filepath)
    rvgaw_post_samples[[b]] <- rvgaw_results$post_samples
  }
}

## Plot posterior densities for different block sizes
param_names <- c("Phi[11]", "Phi[22]", "Sigma[eta[11]]", "Sigma[eta[21]]", "Sigma[eta[22]]")
param_dim <- length(param_names)
# param_values <- c(diag(Phi), Sigma_eta[lower.tri(Sigma_eta, diag = T)])

ind_df <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d)) # (i,j) indices of elements in a dxd matrix

indmat <- matrix(1:d^2, d, d, byrow = T) # number matrix elements by row
phi_indices <- diag(indmat) # indices of diagonal elements of Phi
sigma_indices <- indmat[lower.tri(indmat, diag = T)] # lower triangular elements of Sigma_eta

df_list <- list()
plots <- list()
# rvgaw_post_samples_df <- matrix(NA, nrow = n_post_samples, ncol = length(blocksizes))

for (b in 1:length(blocksizes)) {

  rvgaw.post_samples_Phi <- rvgaw_post_samples[[b]]$Phi
  rvgaw.post_samples_Sigma_eta <- rvgaw_post_samples[[b]]$Sigma_eta

  rvgaw.post_samples_mat <- matrix(NA, nrow = n_post_samples, ncol = param_dim)
  # Arrange posterior samples of Phi in a matrix
  for (k in 1:length(phi_indices)) {
    r <- phi_indices[k]
    i <- as.numeric(ind_df[r, ][1])
    j <- as.numeric(ind_df[r, ][2])
    rvgaw.post_samples_mat[, k] <- sapply(rvgaw.post_samples_Phi, function(x) x[i, j])
    # hmc.post_samples[, k] <- c(hmc.post_samples_Phi[,,r])
    # hmcw.post_samples[, k] <- c(hmcw.post_samples_Phi[,,r])
  }

  # Arrange posterior samples of Sigma_eta in a matrix
  for (k in 1:length(sigma_indices)) {
    r <- sigma_indices[k]
    i <- as.numeric(ind_df[r, ][1])
    j <- as.numeric(ind_df[r, ][2])
    rvgaw.post_samples_mat[, k + d] <- sapply(rvgaw.post_samples_Sigma_eta, function(x) x[i, j])
    # hmc.post_samples[, k+d] <- c(hmc.post_samples_Sigma_eta[,,r])
    # hmcw.post_samples[, k+d] <- c(hmcw.post_samples_Sigma_eta[,,r])
  }

  rvgaw.post_samples_mat <- as.data.frame(rvgaw.post_samples_mat)
  colnames(rvgaw.post_samples_mat) <- param_names
  rvgaw.post_samples_mat$blocksize <- rep(blocksizes[b], n_post_samples)

  df_list[[b]] <- rvgaw.post_samples_mat
}

## Concatenate data frames
concat_df <- do.call(rbind, df_list)

## then do melt here or something
rvgaw_post_samples_df_long <- concat_df %>% pivot_longer(
  cols = !blocksize,
  names_to = "param",
  values_to = "post_samples"
)
rvgaw_post_samples_df_long$blocksize <- factor(rvgaw_post_samples_df_long$blocksize,
  levels = sapply(blocksizes, toString)
)
#   # rvgaw_post_samples_df_long$param <- rep(param_names[p], nrow(rvgaw_post_samples_df_long))

## Plot
# true_vals.df <- data.frame(param = param_names, val = param_values)

plot <- ggplot(rvgaw_post_samples_df_long, aes(x = post_samples)) +
  facet_wrap(vars(param), scales = "free", labeller = label_parsed) +
  geom_density(aes(col = blocksize), lwd = 1.5) +

  # geom_density(data = hmcw.df, col = "goldenrod", lwd = 1) +
  # geom_density(data = hmc.df, col = "deepskyblue", lwd = 1) +

  # geom_vline(
  #   data = true_vals.df, aes(xintercept = val),
  #   color = "black", linetype = "dashed", linewidth = 1
  # ) +
  theme_bw() +
  theme(text = element_text(size = 24)) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 4))

print(plot)


if (save_plots) {
  plot_file <- paste0("compare_blocksizes_multi_sv_real", temper_info, reorder_info,
                      "_", n_indiv, "indiv",
                      "_", transform, "_", date, ".png")
  filepath = paste0("./plots/blocksize_test/", plot_file)
  png(filepath, width = 1200, height = 500)
  # grid.arrange(grobs = plots, nrow = 1, ncol = param_dim)
  print(plot)
  dev.off()
} 

## Plot periodogram

pgram_output <- compute_periodogram(Y_demeaned)
freq <- pgram_output$freq
I <- pgram_output$periodogram

## need to plot the modulus of the periodogram obs here