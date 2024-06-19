# Post-processing results for SV model with simulated data
setwd("~/R-VGA-Whittle/05_Multi_SV_real/")

rm(list = ls())

library(mvtnorm)
library(coda)
# library(Deriv)
# library(cmdstanr)
# library(tensorflow)
# reticulate::use_condaenv("myenv", required = TRUE)
# library(keras)
# library(stats)
# library(bspec)
library(tidyr)
library(dplyr)
library(ggplot2)
library(grid)
library(gridExtra)
library(gtable)

source("./source/compute_periodogram.R")
source("./source/compute_periodogram_uni.R")
source("./source/find_cutoff_freq.R")
source("./source/construct_Sigma.R")

## Flags
# plot_trajectories <- T
save_plots <- T

date <- "20240613" #"20240115" # "20230918" #the 20230918 version has sigma_eta = sqrt(0.1)
use_cholesky <- T # use lower Cholesky factor to parameterise Sigma_eta
transform <- "arctanh"
prior_type <- "prior1"
currencies <- c("GBP", "USD")
use_heaps_mapping <- F
plot_likelihood_surface <- F
plot_prior_samples <- F
plot_trace <- F

## R-VGAW flags
use_tempering <- T #T
temper_first <- T
reorder <- 0 #"decreasing"
reorder_seed <- 2024
# decreasing <- T
use_median <- F

# n_post_samples <- 10000 # per chain
# burn_in <- 5000 # per chain
# n_chains <- 2

## Read data
print("Reading saved data...")
load("./data/exrates.RData")

log_data <- mutate_all(dat, function(x) c(0, log(x[2:length(x)] / x[1:(length(x)-1)]) * 100))

exrates <- log_data[-1, ] # get rid of 1st row
Y <- exrates[, currencies]
Y_demeaned <- Y - colMeans(Y)
d <- ncol(Y_demeaned)

## Read results
print("Reading saved results...")
result_directory <- paste0("./results/forex/", d, "d/", transform, "/")

S <- 1000L
# nblocks <- 100
blocksize <- 100
nsegs <- 25
power_prop <- 1/2

c1 <- find_cutoff_freq(Y_demeaned[, 1], nsegs = nsegs, power_prop = power_prop)$cutoff_ind
c2 <- find_cutoff_freq(Y_demeaned[, 2], nsegs = nsegs, power_prop = power_prop)$cutoff_ind
n_indiv <- max(c1, c2)

if (use_tempering) {
  n_temper <- 5
  K <- 100
  temper_schedule <- rep(1/K, K)
  temper_info <- paste0("_temper", n_temper)
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

if (prior_type == "minnesota") {
  prior_type = ""
} else {
  prior_type = paste0("_", prior_type)
}

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_forex", 
                         "_", paste(currencies, collapse = "_"), 
                         temper_info, reorder_info, block_info, "_", date, ".rds")

hmc_filepath <- paste0(result_directory, "hmc_forex", 
                      "_", paste(currencies, collapse = "_"), 
                       "_", date, ".rds")
                          # "_20240115.rds")

hmcw_filepath <- paste0(result_directory, "hmcw_forex", 
                        "_", paste(currencies, collapse = "_"),
                         "_", date, ".rds")
                        # "_20240115.rds")
rvgaw_results <- readRDS(rvgaw_filepath)
hmc_results <- readRDS(hmc_filepath)
hmcw_results <- readRDS(hmcw_filepath)

rvgaw.Phi <- rvgaw_results$post_samples$Phi
rvgaw.Sigma_eta <- rvgaw_results$post_samples$Sigma_eta

hmc.Phi <- hmc_results$draws[,,1:(d^2)]
hmc.Sigma_eta <- hmc_results$draws[,,(d^2+1):(2*d^2)]

hmcw.Phi <- hmcw_results$draws[,,1:(d^2)]
hmcw.Sigma_eta <- hmcw_results$draws[,,(d^2+1):(2*d^2)]

## HMC and HMCW trace plots
# hmc.phi_mcmc <- mcmc(hmc.phi)
# hmc.sigma_eta_mcmc <- mcmc(hmc.sigma_eta)
# hmcw.phi_mcmc <- mcmc(hmcw.phi)
# hmcw.sigma_eta_mcmc <- mcmc(hmcw.sigma_eta)

########################################
##          Posterior plots           ##
########################################

param_names <- c("Phi[11]", "Phi[22]", "Sigma[eta[11]]", "Sigma[eta[21]]", "Sigma[eta[22]]")
param_dim <- length(param_names)
# param_values <- c(diag(Phi), Sigma_eta[lower.tri(Sigma_eta, diag = T)])

ind_df <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d)) # (i,j) indices of elements in a dxd matrix

indmat <- matrix(1:d^2, d, d, byrow = T) # number matrix elements by row
phi_indices <- diag(indmat) # indices of diagonal elements of Phi
sigma_indices <- indmat[lower.tri(indmat, diag = T)] # lower triangular elements of Sigma_eta

hmc.ESS <- c()
hmc.IF <- c()
hmc.acf <- list()

hmcw.ESS <- c()
hmcw.IF <- c()
hmcw.acf <- list()

n_post_samples <- 10000
n_chains <- 2
thin_interval <- 50
rvgaw.post_samples <- matrix(NA, length(rvgaw.Phi), param_dim)
hmc.post_samples <- matrix(NA, prod(dim(hmc.Phi)[1:2]), param_dim)
hmc_thin.post_samples <- matrix(NA, prod(dim(hmc.Phi)[1:2])/thin_interval, param_dim)
hmcw.post_samples <- matrix(NA, prod(dim(hmcw.Phi)[1:2]), param_dim)

# Arrange posterior samples of Phi in a matrix
acf_lags <- c(0, 1, 5, 10, 20, 50, 100)

for (k in 1:length(phi_indices)) {
  r <- phi_indices[k]
  i <- as.numeric(ind_df[r, ][1])
  j <- as.numeric(ind_df[r, ][2])
  rvgaw.post_samples[, k] <- sapply(rvgaw.Phi, function(x) x[i,j])

  hmc_samples <- mcmc(c(hmc.Phi[,,r]))
  hmcw_samples <- mcmc(c(hmcw.Phi[,,r]))

  ## ACF
  hmc.acf[[k]] <- autocorr(hmc_samples, lags = acf_lags, relative=F)
  hmcw.acf[[k]] <- autocorr(hmcw_samples, lags = acf_lags, relative=F)

  ## Effective Sample Size
  hmc.ESS[k] <- coda::effectiveSize(hmc_samples)
  hmcw.ESS[k] <- coda::effectiveSize(hmcw_samples)
    
  # Compute Inefficiency factor
  hmc.IF[k] <- length(hmc_samples)/hmc.ESS[k]
  hmcw.IF[k] <- length(hmcw_samples)/hmcw.ESS[k]

  ## Thin samples
  hmc.post_samples[, k] <- as.vector(window(hmc_samples, thin = 1))
  hmc_thin.post_samples[, k] <- as.vector(window(hmc_samples, thin = thin_interval))
  hmcw.post_samples[, k] <- as.vector(window(hmcw_samples, thin = 1))
}

# Arrange posterior samples of Sigma_eta in a matrix
for (k in 1:length(sigma_indices)) {
  r <- sigma_indices[k]
  i <- as.numeric(ind_df[r, ][1])
  j <- as.numeric(ind_df[r, ][2])
  rvgaw.post_samples[, k+d] <- sapply(rvgaw.Sigma_eta, function(x) x[i,j])

  hmc_samples <- mcmc(c(hmc.Sigma_eta[,,r]))
  hmcw_samples <- mcmc(c(hmcw.Sigma_eta[,,r]))

  ## ACF
  hmc.acf[[k+d]] <- autocorr(hmc_samples, lags = acf_lags, relative=F)
  hmcw.acf[[k+d]] <- autocorr(hmcw_samples, lags = acf_lags, relative=F)

  ## Effective Sample Size
  hmc.ESS[k+d] <- coda::effectiveSize(hmc_samples)
  hmcw.ESS[k+d] <- coda::effectiveSize(hmcw_samples)
    
  ## Compute Inefficiency factor
  hmc.IF[k+d] <- length(hmc_samples)/hmc.ESS[k+d^2]
  hmcw.IF[k+d] <- length(hmcw_samples)/hmcw.ESS[k+d^2]

  ## Thin samples
  hmc.post_samples[, k+d] <- as.vector(window(hmc_samples))
  hmc_thin.post_samples[, k+d] <- as.vector(window(hmc_samples, thin = thin_interval))
  hmcw.post_samples[, k+d] <- as.vector(window(hmcw_samples, thin = 1))
}

rvgaw.df <- as.data.frame(rvgaw.post_samples)
hmc.df <- as.data.frame(hmc.post_samples)
hmc_thin.df <- as.data.frame(hmc_thin.post_samples)
hmcw.df <- as.data.frame(hmcw.post_samples)
names(rvgaw.df) <- param_names
names(hmc.df) <- param_names
names(hmc_thin.df) <- param_names
names(hmcw.df) <- param_names

plots <- list()

xlims <- list(c(0.985, 1), c(0.985, 1), c(0, 0.02), c(0, 0.02), c(0, 0.02))

## Marginal posteriors
for (p in 1:param_dim) {
  
  # true_vals.df <- data.frame(name = param_names[p], val = param_values[p])


  plot <- ggplot(rvgaw.df, aes(x=.data[[param_names[p]]])) +
    # plot <- ggplot(exact_rvgal.df, aes(x=colnames(exact_rvgal.df)[p])) + 
    geom_density(col = "red", lwd = 1) +
    geom_density(data = hmcw.df, col = "goldenrod", lwd = 1) +
    geom_density(data = hmc_thin.df, col = "deepskyblue", lwd = 1) +
    # geom_vline(data = true_vals.df, aes(xintercept=val),
    #            color="black", linetype="dashed", linewidth=1) +
    labs(x = vars) +
    # xlim(x = xlims[[p]]) +
    theme_bw() +
    theme(axis.title = element_blank(), text = element_text(size = 24)) +
    scale_x_continuous(limits = xlims[[p]], breaks = scales::pretty_breaks(n = 3)) + 
    theme(plot.margin = margin(0.3, 0.3, 0.3, 0.3, "cm"))
  
  plots[[p]] <- plot  
}

## Arrange bivariate plots in lower off-diagonals
n_lower_tri <- (param_dim^2 - param_dim)/2 # number of lower triangular elements

index_to_i_j_colwise_nodiag <- function(k, n) {
  kp <- n * (n - 1) / 2 - k
  p  <- floor((sqrt(1 + 8 * kp) - 1) / 2)
  i  <- n - (kp - p * (p + 1) / 2)
  j  <- n - 1 - p
  c(i, j)
}

cov_plots <- list()
for (ind in 1:n_lower_tri) {
  mat_ind <- index_to_i_j_colwise_nodiag(ind, param_dim)
  p <- mat_ind[1]
  q <- mat_ind[2]
  
  # param_df <- data.frame(x = param_values[q], y = param_values[p])

  cov_plot <- ggplot(rvgaw.df, aes(x = .data[[param_names[q]]], y = .data[[param_names[p]]])) +
    stat_ellipse(col = "red", type = "norm", lwd = 1) +
    stat_ellipse(data = hmcw.df, col = "goldenrod", type = "norm", lwd = 1) +
    stat_ellipse(data = hmc.df, col = "deepskyblue", type = "norm", lwd = 1) +
    theme_bw() +
    theme(axis.title = element_blank(), text = element_text(size = 24)) +                               # Assign pretty axis ticks
    scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) +
    theme(plot.margin = margin(0.35, 0.35, 0.35, 0.35, "cm"))

  cov_plots[[ind]] <- cov_plot
}

m <- matrix(NA, param_dim, param_dim)
m[lower.tri(m, diag = F)] <- 1:n_lower_tri 
gr <- grid.arrange(grobs = cov_plots, layout_matrix = m)
gr2 <- gtable_add_cols(gr, unit(1, "null"), -1)
gr3 <- gtable_add_grob(gr2, grobs = lapply(plots, ggplotGrob), t = 1:param_dim, l = 1:param_dim)

# grid.draw(gr3)

# A list of text grobs - the labels
vars <- list(textGrob(bquote(Phi[11])), textGrob(bquote(Phi[22])),
             textGrob(bquote(Sigma[eta[11]])), textGrob(bquote(Sigma[eta[21]])),
             textGrob(bquote(Sigma[eta[21]])))
vars <- lapply(vars, editGrob, gp = gpar(col = "black", fontsize = 24))

# m <- matrix(1:param_dim, 1, param_dim, byrow = T)
# gr <- grid.arrange(grobs = plots, layout_matrix = m)
# gp <- gtable_add_rows(gr, unit(1.5, "lines"), -1) #0 adds on the top
# gtable_show_layout(gp)
# 
# gp <- gtable_add_grob(gp, vars[1:param_dim], t = 2, l = 1:3)

# So that there is space for the labels,
# add a row to the top of the gtable,
# and a column to the left of the gtable.
gp <- gtable_add_cols(gr3, unit(2, "lines"), 0)
gp <- gtable_add_rows(gp, unit(2, "lines"), -1) #0 adds on the top

# gtable_show_layout(gp)

# Add the label grobs.
# The labels on the left should be rotated; hence the edit.
# t and l refer to cells in the gtable layout.
# gtable_show_layout(gp) shows the layout.
gp <- gtable_add_grob(gp, lapply(vars[1:param_dim], editGrob, rot = 90), t = 1:param_dim, l = 1)
gp <- gtable_add_grob(gp, vars[1:param_dim], t = param_dim+1, l = 2:(param_dim+1))

grid.newpage()
grid.draw(gp)

if (save_plots) {
  plot_file <- paste0("multi_sv_real_posterior_", 
                      paste(currencies, collapse = "_"),
                      temper_info, reorder_info, block_info,
                      "_", transform, "_thinned_", date, ".png")
  filepath = paste0("./plots/", plot_file)
  png(filepath, width = 1200, height = 900)
  grid.draw(gp)
  dev.off()
}

## R-VGA-Whittle trajectories/trace plots
# if (plot_trajectories) {

  # param_names <- c("Phi[11]", "Phi[22]", "Sigma[eta[11]]", "Sigma[eta[22]]", "Sigma[eta[21]]")
  # param_vals <- c(diag(Phi), diag(Sigma_eta), Sigma_eta[lower.tri(Sigma_eta, diag = F)])
  # true_vals.df <- data.frame(param = param_names, 
  #                           value = param_values)

  
  mu_Phi <- lapply(rvgaw_results$mu, function(x) tanh(x[1:d]))
  mu_Sigma <- lapply(rvgaw_results$mu, construct_Sigma_eta, d = d)
  mu_Sigma_vec <- lapply(mu_Sigma, function(S) S[lower.tri(S, diag = T)])
  mu <- mapply(c, mu_Phi, mu_Sigma_vec, SIMPLIFY = F)

  block_df <- data.frame(cutoff = n_indiv)

  trajectory_df <- as.data.frame(matrix(unlist(mu), nrow = length(mu), byrow = T))
  names(trajectory_df) <- param_names
  
    trajectory_df$iter <- 1:nrow(trajectory_df)

    trajectory_df_long <- trajectory_df %>% pivot_longer(
        cols = !iter,
        names_to = "param", values_to = "value"
    )

    trajectory_plot <- trajectory_df_long %>% ggplot() +
        geom_line(aes(x = iter, y = value), linewidth = 1) +
        facet_wrap(~param, scales = "free", labeller = label_parsed) +
        # geom_hline(data = true_vals.df, aes(yintercept = value), linetype = "dashed", linewidth = 1.5) +
        geom_vline(data = block_df, aes(xintercept = cutoff), linetype = "dotted", linewidth = 1.5) +
        theme_bw() +
        theme(text = element_text(size = 28)) +
        xlab("Iterations") +
        ylab("Value")

    if (save_plots) {
      png(paste0("plots/trajectories_multi_sv_real", block_info, ".png"), width = 1000, height = 500)
      print(trajectory_plot)
      dev.off()
    }

# }

## HMC and HMC-Whittle trace plots

  hmc.df_long <- hmc.df %>% 
      mutate(n = row_number()) %>% 
      pivot_longer(
          cols = !n,
          names_to = "param", values_to = "value"
      )

  hmc_thin.df_long <- hmc_thin.df %>% 
  mutate(n = row_number()) %>% 
  pivot_longer(
      cols = !n,
      names_to = "param", values_to = "value"
  )

  hmcw.df_long <- hmcw.df %>% mutate(n = row_number()) %>% 
  pivot_longer(
      cols = !n,
      names_to = "param", values_to = "value"
  )

## Traceplots

# hmc.traceplots <- list()
# for (p in 1:param_dim) {
# p <- 1
  hmc.traceplot <- hmc.df_long %>% ggplot() + geom_line(aes(x = n, y = value), linewidth = 1) +
      # geom_hline(data = true_vals.df, aes(yintercept = value), col = "red", 
      #             linetype = "dashed", linewidth = 1.5) +
      facet_wrap(~param, scales = "free", labeller = label_parsed) +
      theme_bw() +
      theme(text = element_text(size = 28)) +
      xlab("Iterations") +
      ylab("Value")
  print(hmc.traceplot)

  hmc_thin.traceplot <- hmc_thin.df_long %>% ggplot() + geom_line(aes(x = n, y = value), linewidth = 1) +
  # geom_hline(data = true_vals.df, aes(yintercept = value), col = "red", 
  #             linetype = "dashed", linewidth = 1.5) +
  facet_wrap(~param, scales = "free", labeller = label_parsed) +
  theme_bw() +
  theme(text = element_text(size = 28)) +
  xlab("Iterations") +
  ylab("Value")
  print(hmc_thin.traceplot)

  hmcw.traceplot <- hmcw.df_long %>% ggplot() + geom_line(aes(x = n, y = value), linewidth = 1) +
  # geom_hline(data = true_vals.df, aes(yintercept = value), col = "red", 
  #             linetype = "dashed", linewidth = 1.5) +
  facet_wrap(~param, scales = "free", labeller = label_parsed) +
  theme_bw() +
  theme(text = element_text(size = 28)) +
  xlab("Iterations") +
  ylab("Value")
  print(hmcw.traceplot)
    # scale_x_continuous(breaks = scales::pretty_breaks(n = 3))
  # theme(legend.position="bottom") + 
  # scale_color_manual(values = c('RVGA' = 'red', 'HMC' = 'blue'))

  # hmc.traceplots[[p]] <- plot  
# }
  if (save_plots)  {
      png("./plots/multi_sv_real_hmc_traceplot.png", width = 1500, height = 500)
      print(hmc.traceplot)
      dev.off()

      png("./plots/multi_sv_real_hmc_traceplot_thin.png", width = 1500, height = 500)
      print(hmc_thin.traceplot)
      dev.off()

      png("./plots/multi_sv_real_hmcw_traceplot.png", width = 1500, height = 500)
      print(hmcw.traceplot)
      dev.off()

  }


## Timing comparison
rvgaw.time <- rvgaw_results$time_elapsed[3]
hmcw.time <- sum(hmcw_results$time()$chains$total)
hmc.time <- sum(hmc_results$time()$chains$total)
print(data.frame(method = c("R-VGA-Whittle", "HMC-Whittle", "HMC"),
                 time = c(rvgaw.time, hmcw.time, hmc.time)))
