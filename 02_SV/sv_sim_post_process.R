# Post-processing results for SV model with simulated data
setwd("~/R-VGA-Whittle/02_SV/")

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
# library(tidyr)
library(dplyr)
library(ggplot2)
library(grid)
library(gridExtra)
library(gtable)

source("./source/compute_periodogram.R")
source("./source/find_cutoff_freq.R")


## Flags
date <- "20240214" # "20230918" #the 20230918 version has sigma_eta = sqrt(0.1)
# date <- "20230918"

## R-VGA flags
# regenerate_data <- F
# save_data <- F
use_tempering <- T
temper_first <- T
reorder <- 0 # "decreasing" # or decreasing # or a number
reorder_seed <- 2024
plot_prior <- F
plot_likelihood_surface <- F
prior_type <- ""
transform <- "arctanh"
plot_trajectories <- T
save_plots <- T

# n_post_samples <- 10000 # per chain
# burn_in <- 5000 # per chain
# n_chains <- 2

n <- 10000
phi <- 0.99

## Read data
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number
print("Reading saved data...")
sv_data <- readRDS(file = paste0("./data/sv_data_n", n, "_phi", phi_string, "_", date, ".rds"))

y <- sv_data$y
x <- sv_data$x
phi <- sv_data$phi
sigma_eta <- sv_data$sigma_eta
sigma_eps <- sv_data$sigma_eps

## Read results
print("Reading saved results...")
result_directory <- paste0("./results/", transform, "/")

S <- 1000L
# nblocks <- 100
blocksize <- 100
n_indiv <- find_cutoff_freq(y, nsegs = 25, power_prop = 1 / 2)$cutoff_ind # 100

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

# if (!is.null(nblocks)) {
if (!is.null(blocksize)) {
    # block_info <- paste0("_", nblocks, "blocks", n_indiv, "indiv")
    block_info <- paste0("_", "blocksize", blocksize, "_", n_indiv, "indiv")
} else {
    block_info <- ""
}

rvgaw_filepath <- paste0(
    result_directory, "rvga_whittle_results_n", n,
    "_phi", phi_string, temper_info, reorder_info, block_info,
    prior_type, "_", date, ".rds"
)
hmc_filepath <- paste0(
    result_directory, "hmc_results_n", n,
    "_phi", phi_string, "_", date, ".rds"
)
hmcw_filepath <- paste0(
    result_directory, "hmcw_results_n", n,
    "_phi", phi_string, "_", date, ".rds"
)

rvgaw_results <- readRDS(rvgaw_filepath)
hmc_results <- readRDS(hmc_filepath)
hmcw_results <- readRDS(hmcw_filepath)

rvgaw.phi <- rvgaw_results$post_samples$phi
rvgaw.sigma_eta <- rvgaw_results$post_samples$sigma_eta

hmc.phi <- c(hmc_results$draws[, , 1]) # tanh(hmc.theta_phi)
hmc.sigma_eta <- c(hmc_results$draws[, , 2]) # sqrt(exp(hmc.theta_sigma))

hmcw.phi <- c(hmcw_results$draws[, , 1])
hmcw.sigma_eta <- c(hmcw_results$draws[, , 2])

param_names <- c("phi", "sigma[eta]")
param_dim <- length(param_names)

## HMC and HMCW trace plots
hmc.phi_mcmc <- mcmc(hmc.phi)
hmc.sigma_eta_mcmc <- mcmc(hmc.sigma_eta)
hmcw.phi_mcmc <- mcmc(hmcw.phi)
hmcw.sigma_eta_mcmc <- mcmc(hmcw.sigma_eta)

## Estimation of kappa
# mean_log_eps2 <- digamma(1/2) + log(2)
# log_kappa2 <- mean(log(y^2)) - mean_log_eps2
# kappa <- sqrt(exp(log_kappa2))

########################################
##          Posterior plots           ##
########################################

hmc.ESS <- c()
hmc.IF <- c()
hmc.acf <- list()

hmcw.ESS <- c()
hmcw.IF <- c()
hmcw.acf <- list()


## ACF, ESS and inefficiency factor
hmc.acf[[1]] <- autocorr(hmc.phi_mcmc, lags = c(0, 1, 5, 10, 20, 50, 100), relative = F)
hmc.acf[[2]] <- autocorr(hmc.sigma_eta_mcmc, lags = c(0, 1, 5, 10, 20, 50, 100), relative = F)

hmc.ESS[1] <- effectiveSize(hmc.phi_mcmc)
hmc.ESS[2] <- effectiveSize(hmc.sigma_eta_mcmc)
hmc.IF[1] <- length(hmc.phi_mcmc) / hmc.ESS[1]
hmc.IF[2] <- length(hmc.sigma_eta_mcmc) / hmc.ESS[2]

hmcw.acf[[1]] <- autocorr(hmcw.phi_mcmc, lags = c(0, 1, 5, 10, 20, 50, 100), relative = F)
hmcw.acf[[2]] <- autocorr(hmcw.sigma_eta_mcmc, lags = c(0, 1, 5, 10, 20, 50, 100), relative = F)

hmcw.ESS[1] <- effectiveSize(hmcw.phi_mcmc)
hmcw.ESS[2] <- effectiveSize(hmcw.sigma_eta_mcmc)
hmcw.IF[1] <- length(hmcw.phi_mcmc) / hmcw.ESS[1]
hmcw.IF[2] <- length(hmcw.sigma_eta_mcmc) / hmcw.ESS[2]

## Thinning
thin_interval <- 100
hmc.phi_thin <- as.vector(window(hmc.phi_mcmc, thin = thin_interval))
hmc.sigma_eta_thin <- as.vector(window(hmc.sigma_eta_mcmc, thin = thin_interval))

hmcw.phi_thin <- as.vector(window(hmcw.phi_mcmc, thin = 1))
hmcw.sigma_eta_thin <- as.vector(window(hmcw.sigma_eta_mcmc, thin = 1))

rvgaw.df <- data.frame(
    phi = rvgaw.phi,
    sigma_eta = rvgaw.sigma_eta
)

hmc.df <- data.frame(
    phi = hmc.phi,
    sigma_eta = hmc.sigma_eta
)

hmc_thin.df <- data.frame(
    phi = hmc.phi_thin,
    sigma_eta = hmc.sigma_eta_thin
)
hmcw.df <- data.frame(
    phi = hmcw.phi_thin,
    sigma_eta = hmcw.sigma_eta_thin
)

names(rvgaw.df) <- param_names
names(hmc.df) <- param_names
names(hmc_thin.df) <- param_names
names(hmcw.df) <- param_names

# true_vals.df <- data.frame(phi = phi, sigma_eta = sigma_eta)

## Posterior plots
param_values <- c(phi, sigma_eta)

plots <- list()

for (p in 1:param_dim) {
    true_vals.df <- data.frame(name = param_names[p], val = param_values[p])

    plot <- ggplot(rvgaw.df, aes(x = .data[[param_names[p]]])) +
        geom_density(col = "red", lwd = 1) +
        geom_density(data = hmcw.df, col = "goldenrod", lwd = 1) +
        geom_density(data = hmc_thin.df, col = "deepskyblue", lwd = 1) +
        geom_vline(
            data = true_vals.df, aes(xintercept = val),
            color = "black", linetype = "dashed", linewidth = 1
        ) +
        labs(x = vars) +
        theme_bw() +
        theme(axis.title = element_blank(), text = element_text(size = 24)) +
        scale_x_continuous(breaks = scales::pretty_breaks(n = 4))

    plots[[p]] <- plot
}

## Arrange bivariate plots in lower off-diagonals
n_lower_tri <- (param_dim^2 - param_dim) / 2 # number of lower triangular elements

index_to_i_j_colwise_nodiag <- function(k, n) {
    kp <- n * (n - 1) / 2 - k
    p <- floor((sqrt(1 + 8 * kp) - 1) / 2)
    i <- n - (kp - p * (p + 1) / 2)
    j <- n - 1 - p
    c(i, j)
}

cov_plots <- list()
for (ind in 1:n_lower_tri) {
    mat_ind <- index_to_i_j_colwise_nodiag(ind, param_dim)
    p <- mat_ind[1]
    q <- mat_ind[2]

    param_df <- data.frame(x = param_values[q], y = param_values[p])

    cov_plot <- ggplot(rvgaw.df, aes(x = .data[[param_names[q]]], y = .data[[param_names[p]]])) +
        stat_ellipse(col = "red", type = "norm", lwd = 1) +
        stat_ellipse(data = hmcw.df, col = "goldenrod", type = "norm", lwd = 1) +
        stat_ellipse(data = hmc.df, col = "deepskyblue", type = "norm", lwd = 1) +
        geom_point(
            data = param_df, aes(x = x, y = y),
            shape = 4, color = "black", size = 5
        ) +
        theme_bw() +
        theme(axis.title = element_blank(), text = element_text(size = 24)) + # Assign pretty axis ticks
        scale_x_continuous(breaks = scales::pretty_breaks(n = 3))

    cov_plots[[ind]] <- cov_plot
}

m <- matrix(NA, param_dim, param_dim)
m[lower.tri(m, diag = F)] <- 1:n_lower_tri
gr <- grid.arrange(grobs = cov_plots, layout_matrix = m)
gr2 <- gtable_add_cols(gr, unit(1, "null"), -1)
gr3 <- gtable_add_grob(gr2, grobs = lapply(plots, ggplotGrob), t = 1:param_dim, l = 1:param_dim)

grid.draw(gr3)

# A list of text grobs - the labels
vars <- list(textGrob(bquote(phi)), textGrob(bquote(sigma[eta])), textGrob(bquote(sigma[epsilon])))
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
gp <- gtable_add_cols(gr3, unit(1.5, "lines"), 0)
gp <- gtable_add_rows(gp, unit(1.5, "lines"), -1) # 0 adds on the top

gtable_show_layout(gp)

# Add the label grobs.
# The labels on the left should be rotated; hence the edit.
# t and l refer to cells in the gtable layout.
# gtable_show_layout(gp) shows the layout.
gp <- gtable_add_grob(gp, lapply(vars[1:param_dim], editGrob, rot = 90), t = 1:param_dim, l = 1)
gp <- gtable_add_grob(gp, vars[1:param_dim], t = param_dim + 1, l = 2:(param_dim + 1))

grid.newpage()
grid.draw(gp)

if (save_plots) {
    plot_file <- paste0(
        "sv_sim_posterior", "_", n, temper_info, reorder_info, block_info,
        "_", transform, "_thinned_", date, ".png"
    )
    filepath <- paste0("./plots/", plot_file)
    png(filepath, width = 800, height = 600)
    grid.draw(gp)
    dev.off()
}


## Timing comparison
rvgaw.time <- rvgaw_results$time_elapsed[3]
hmcw.time <- sum(hmcw_results$time()$chains$total)
hmc.time <- sum(hmc_results$time()$chains$total)
print(data.frame(
    method = c("R-VGA-Whittle", "HMC-Whittle", "HMC-exact"),
    time = c(rvgaw.time, hmcw.time, hmc.time)
))

## Trajectories/Trace plots
if (plot_trajectories) {
    mu_phi <- sapply(rvgaw_results$mu, function(x) x[1])
    mu_sigma_eta <- sapply(rvgaw_results$mu, function(x) x[2])

    if (transform == "arctanh") {
        mu_phi <- tanh(mu_phi)
    } else { # logit transform
        mu_phi <- exp(mu_phi) / (1 + exp(mu_phi))
    }
    mu_sigma_eta <- sqrt(exp(mu_sigma_eta))

    true_df <- data.frame(
        param = c("phi", "sigma[eta]"),
        value = c(phi, sigma_eta)
    )

    block_df <- data.frame(cutoff = n_indiv)

    trajectory_df <- data.frame(phi = mu_phi, sigma_eta = mu_sigma_eta)
    names(trajectory_df) <- c("phi", "sigma[eta]")
    trajectory_df$iter <- 1:nrow(trajectory_df)

    trajectory_df_long <- trajectory_df %>% pivot_longer(
        cols = !iter,
        names_to = "param", values_to = "value"
    )
    trajectory_plot <- trajectory_df_long %>% ggplot() +
        geom_line(aes(x = iter, y = value), linewidth = 1) +
        facet_wrap(~param, scales = "free", labeller = label_parsed) +
        geom_hline(data = true_df, aes(yintercept = value), linetype = "dashed", linewidth = 1.5) +
        geom_vline(data = block_df, aes(xintercept = cutoff), linetype = "dotted", linewidth = 1.5) +
        theme_bw() +
        theme(text = element_text(size = 34)) +
        xlab("Iterations") +
        ylab("Value")

    png(paste0("plots/trajectories_sv_sim", block_info, ".png"), width = 1200, height = 500)
    print(trajectory_plot)

    dev.off()

    par(mfrow = c(2,1))
    # coda::traceplot(hmc.phi_mcmc, density = T)
    # coda::traceplot(hmc.sigma_eta_mcmc, density = T)
    # coda::traceplot(hmcw.phi_mcmc, density = T)
    # coda::traceplot(hmcw.sigma_eta_mcmc, density = T)
    
    # true_df <- data.frame(
    #     param = c("phi", "sigma[eta]"),
    #     value = c(phi, sigma_eta)
    # )

    # hmc.df <- data.frame(
    # phi = hmc.phi,
    # sigma_eta = hmc.sigma_eta
    # )
    # names(hmc.df) <- param_names

    # hmcw.df <- data.frame(
    #     phi = hmcw.phi,
    #     sigma_eta = hmcw.sigma_eta
    # )
    # names(hmcw.df) <- param_names    
    
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
    hmc.traceplot <- hmc.df_long %>% ggplot() + geom_line(aes(x = n, y = value), linewidth = 1) +
        geom_hline(data = true_df, aes(yintercept = value), col = "red", 
                    linetype = "dashed", linewidth = 1.5) +
        facet_wrap(~param, scales = "free", labeller = label_parsed) +
        theme_bw() +
        theme(text = element_text(size = 28)) +
        xlab("Iterations") +
        ylab("Value")

    print(hmc.traceplot)

     hmc_thin.traceplot <- hmc_thin.df_long %>% ggplot() + geom_line(aes(x = n, y = value), linewidth = 1) +
        geom_hline(data = true_df, aes(yintercept = value), col = "red", 
                    linetype = "dashed", linewidth = 1.5) +
        facet_wrap(~param, scales = "free", labeller = label_parsed) +
        theme_bw() +
        theme(text = element_text(size = 28)) +
        xlab("Iterations") +
        ylab("Value")

    print(hmc_thin.traceplot)

    hmcw.traceplot <- hmcw.df_long %>% ggplot() + geom_line(aes(x = n, y = value), linewidth = 1) +
        geom_hline(data = true_df, aes(yintercept = value), col = "red", 
                    linetype = "dashed", linewidth = 1.5) +
        facet_wrap(~param, scales = "free", labeller = label_parsed) +
        theme_bw() +
        theme(text = element_text(size = 28)) +
        xlab("Iterations") +
        ylab("Value")

    print(hmcw.traceplot)

    if (save_plots)  {
        png("./plots/sv_sim_hmc_traceplot.png", width = 1500, height = 500)
        print(hmc.traceplot)
        dev.off()

        png("./plots/sv_sim_hmc_traceplot_thin.png", width = 1500, height = 500)
        print(hmc_thin.traceplot)
        dev.off()

        png("./plots/sv_sim_hmcw_traceplot.png", width = 1500, height = 500)
        print(hmcw.traceplot)
        dev.off()

    }

}