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
library(tidyr)
library(ggplot2)
library(grid)
library(gtable)
library(gridExtra)

# source("./source/run_rvgaw_lgss_tf.R")
source("./source/run_rvgaw_lgss_block.R")
source("./source/run_mcmc_lgss_allparams.R")
source("./source/compute_periodogram.R")
source("./source/compute_kf_likelihood.R")
source("./source/compute_whittle_likelihood_lgss.R")
source("./source/update_sigma.R")
source("./source/run_hmc_lgss.R")
source("./source/find_cutoff_freq.R")

################## Some code to limit tensorflow memory usage ##################

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

################## End of code to limit tensorflow memory usage ##################

## Result directory
result_directory <- "./results/"

## Flags
date <- "20230525"
# date <- "20230302"

regenerate_data <- F
save_data <- F

rerun_rvgaw <- T
rerun_mcmcw <- F
rerun_hmc <- F
rerun_hmcw <- F

save_rvgaw_results <- T
save_mcmcw_results <- F
save_hmc_results <- F
save_hmcw_results <- F

plot_likelihood_surface <- F
plot_trajectories <- F
save_plots <- F

## R-VGA flags
use_tempering <- T
temper_first <- T
reorder <- 0 #"decreasing"
reorder_seed <- 2024
# decreasing <- T
transform <- "arctanh"

## MCMC flags
adapt_proposal <- T
n_post_samples <- 20000
burn_in <- 10000

## True parameters
sigma_eps <- 0.5 # measurement error var
sigma_eta <- 0.7 # process error var
phi <- 0.9

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

n <- 10000

if (regenerate_data) {
  print("Generating data...")
  
  # Generate true process x_0:T
  x <- c()
  set.seed(2023)
  x0 <- rnorm(1, 0, sqrt(sigma_eta^2 / (1-phi^2)))
  x[1] <- x0
  for (t in 2:(n+1)) {
    x[t] <- phi * x[t-1] + rnorm(1, 0, sigma_eta)
  }
  
  # Generate observations y_1:T
  y <- x[2:(n+1)] + rnorm(n, 0, sigma_eps)
  
  ## Plot true process and observations
  # par(mfrow = c(1, 1))
  # plot(x, type = "l", main = "True process")
  # points(y, col = "cyan")
  
  lgss_data <- list(x = x, y = y, phi = phi, sigma_eps = sigma_eps, sigma_eta = sigma_eta)
  if (save_data) {
    saveRDS(lgss_data, file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  }
} else {
  print("Reading saved data...")
  lgss_data <- readRDS(file = paste0("./data/lgss_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  
}

y <- lgss_data$y
x <- lgss_data$x
phi <- lgss_data$phi
sigma_eps <- lgss_data$sigma_eps
sigma_eta <- lgss_data$sigma_eta

## MCMC settings
n_post_samples <- 20000
burn_in <- 10000
MCMC_iters <- n_post_samples + burn_in # Number of MCMC iterations

## Prior
prior_mean <- c(0, -1, -1)#rep(0, 3)
prior_var <- diag(c(1, 1, 1))

## Initial state mean and variance for the KF
# state_ini_mean <- 0
# state_ini_var <- 1

if (plot_likelihood_surface) {
## Test the likelihood computation by plotting likelihood surface over a grid of parameter values
phi_grid <- seq(-1, 1, length.out = 200)
likelihood_whittle <- c()
likelihood_exact <- c()

for (i in 1:length(phi_grid)) {
  params_list <- list(phi = phi_grid[i], sigma_eta = sigma_eta, sigma_eps = sigma_eps)
  # if (use_whittle_likelihood) {
  likelihood_whittle[i] <- compute_whittle_likelihood_lgss(y = y, params = params_list)
  # } else {
  kf_out <- compute_kf_likelihood(state_prior_mean = state_ini_mean,
                                  state_prior_var = state_ini_var,
                                  iters = length(y), observations = y,
                                  params = params_list)
  
  likelihood_exact[i] <- kf_out$log_likelihood
  # }
}
  par(mfrow = c(2,1))
  margin <- 20
  plot_range <- (which.max(likelihood_exact) - margin):(which.max(likelihood_exact) + margin)
  plot(phi_grid[plot_range], likelihood_exact[plot_range], type = "l",
      xlab = "phi", ylab = "log likelihood", main = paste0("Exact likelihood (n = ", n, ")"))
  legend("topleft", legend = c("true value", "arg max llh"),
        col = c("black", "red"), lty = 2, cex = 0.5)
  abline(v = phi_grid[which.max(likelihood_exact)], lty = 1, col = "red")
  abline(v = phi, lty = 2)

  plot(phi_grid[plot_range], likelihood_whittle[plot_range], type = "l",
      xlab = "phi", ylab = "log likelihood", main = paste0("Whittle likelihood (n = ", n, ")"))
  legend("topleft", legend = c("true value", "arg max llh"),
        col = c("black", "red"), lty = 2, cex = 0.5)
  abline(v = phi_grid[which.max(likelihood_whittle)], lty = 1, col = "red")
  abline(v = phi, lty = 2)

  browser()
}

## Sample from prior to see if values of phi are reasonable
# theta_samples <- rnorm(10000, prior_mean, prior_var)
# if (transform == "arctanh") {
#   phi_samples <- tanh(theta_samples)
# } else {
#   phi_samples <- exp(theta_samples) / (1 + exp(theta_samples))
# }
# 
# hist(phi_samples, main = "Samples from the prior of phi")

###############################################################################
##                      R-VGA with Whittle likelihood                         ##
################################################################################

S <- 1000L

# nblocks <- 100
n_indiv <- find_cutoff_freq(y, nsegs = 25, power_prop = 1/2)$cutoff_ind #500 #220 #1000 #807
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

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_", transform, "_n", n,
                         "_phi", phi_string, temper_info, reorder_info, block_info, "_", date, ".rds")

if (rerun_rvgaw) {
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
  
} else {
  rvgaw_results <- readRDS(rvgaw_filepath)
}

rvgaw.post_samples_phi <- rvgaw_results$post_samples$phi
rvgaw.post_samples_eta <- rvgaw_results$post_samples$sigma_eta
rvgaw.post_samples_eps <- rvgaw_results$post_samples$sigma_eps

################################################################################
##                       MCMC with Whittle likelihood                         ##
################################################################################

mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_results_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

if (rerun_mcmcw) {
  mcmcw_results <- run_mcmc_lgss(y, #sigma_eta, sigma_eps, 
                                 iters = MCMC_iters, burn_in = burn_in,
                                 prior_mean = prior_mean, prior_var = prior_var,  
                                 state_ini_mean = state_ini_mean, state_ini_var = state_ini_var,
                                 adapt_proposal = T, use_whittle_likelihood = T)
  
  if (save_mcmcw_results) {
    saveRDS(mcmcw_results, mcmcw_filepath)
  }
} else {
  mcmcw_results <- readRDS(mcmcw_filepath)
}

mcmcw.post_samples_phi <- as.mcmc(mcmcw_results$post_samples$phi[-(1:burn_in)])
mcmcw.post_samples_eta <- as.mcmc(mcmcw_results$post_samples$sigma_eta[-(1:burn_in)])
mcmcw.post_samples_eps <- as.mcmc(mcmcw_results$post_samples$sigma_eps[-(1:burn_in)])

##################################################
###        HMC with the exact likelihood       ###
##################################################

hmc_filepath <- paste0(result_directory, "hmc_results_n", n, 
                       "_phi", phi_string, "_", date, ".rds")

# n_post_samples <- 10000
# burn_in <- 1000
# stan.iters <- n_post_samples + burn_in
n_chains <- 2
if (rerun_hmc) {
  hmc_results <- run_hmc_lgss(data = y, 
                                iters = n_post_samples / n_chains, 
                                burn_in = burn_in / n_chains)
  
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

hmc.post_samples_phi <- hmc_results$draws[,,1]#tanh(hmc.theta_phi)
hmc.post_samples_eta <- hmc_results$draws[,,2]#sqrt(exp(hmc.theta_sigma))
hmc.post_samples_eps <- hmc_results$draws[,,3]#sqrt(exp(hmc.theta_sigma))

#######################################################
##          HMC with the Whittle likelihood          ##
#######################################################
hmcw_filepath <- paste0(result_directory, "hmcw_results_n", n, 
                       "_phi", phi_string, "_", date, ".rds")

if (rerun_hmcw) {
  
  # ## Fourier frequencies
  # k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
  # k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
  # freq <- 2 * pi * k_in_likelihood / n
  
  # ## Fourier transform of the observations
  # fourier_transf <- fft(y)
  # periodogram <- 1/n * Mod(fourier_transf)^2
  # I <- periodogram[k_in_likelihood + 1]

  pgram_output <- compute_periodogram(y)
  freq <- pgram_output$freq
  I <- pgram_output$periodogram
  
  whittle_stan_file <- "./source/stan_lgss_whittle.stan"
  
  whittle_lgss_model <- cmdstan_model(
    whittle_stan_file,
    cpp_options = list(stan_threads = TRUE)
  )
  
  # log_kappa2_est <- mean(log(y^2)) - (digamma(1/2) + log(2))
  
  whittle_lgss_data <- list(nfreq = length(freq), freqs = freq, periodogram = I)
  
  fit_stan_lgss_whittle <- whittle_lgss_model$sample(
    whittle_lgss_data,
    chains = n_chains,
    threads = parallel::detectCores(),
    refresh = 100,
    iter_warmup = burn_in / n_chains,
    iter_sampling = n_post_samples / n_chains
  )
  
  hmcw_results <- list(draws = fit_stan_lgss_whittle$draws(variables = c("phi", "sigma_eta", "sigma_eps")),
                       time = fit_stan_lgss_whittle$time,
                       summary = fit_stan_lgss_whittle$cmdstan_summary)
  # fit_stan_lgss_whittle$cmdstan_summary()
  # fit_stan_lgss_whittle$diagnostic_summary()
  
  if (save_hmcw_results) {
    saveRDS(hmcw_results, hmcw_filepath)
  }
  
} else {
  hmcw_results <- readRDS(hmcw_filepath)
}

hmcw.post_samples_phi <- c(hmcw_results$draws[,,1])
hmcw.post_samples_eta <- c(hmcw_results$draws[,,2])
hmcw.post_samples_eps <- c(hmcw_results$draws[,,3])

################################################################################
##                            Posterior densities                             ##
################################################################################

hmc.post_samples_phi_mcmc <- mcmc(c(hmc.post_samples_phi))
hmc.post_samples_sigma_eta_mcmc <- mcmc(c(hmc.post_samples_eta))
hmc.post_samples_sigma_eps_mcmc <- mcmc(c(hmc.post_samples_eps))

## ACF, ESS and inefficiency factor
hmc.acf <- list()
hmc.ESS <- c()
hmc.IF <- c()

hmc.acf[[1]] <- autocorr(hmc.post_samples_phi_mcmc, lags = c(0, 1, 5, 10, 20, 50, 100), relative=F)
hmc.acf[[2]] <- autocorr(hmc.post_samples_sigma_eta_mcmc, lags = c(0, 1, 5, 10, 20, 50, 100), relative=F)
hmc.acf[[3]] <- autocorr(hmc.post_samples_sigma_eps_mcmc, lags = c(0, 1, 5, 10, 20, 50, 100), relative=F)

hmc.ESS[1] <- effectiveSize(hmc.post_samples_phi_mcmc)
hmc.ESS[2] <- effectiveSize(hmc.post_samples_sigma_eta_mcmc)
hmc.ESS[3] <- effectiveSize(hmc.post_samples_sigma_eps_mcmc)

hmc.IF[1] <- length(hmc.post_samples_phi_mcmc)/hmc.ESS[1]
hmc.IF[2] <- length(hmc.post_samples_sigma_eta_mcmc)/hmc.ESS[2]
hmc.IF[3] <- length(hmc.post_samples_sigma_eps_mcmc)/hmc.ESS[3]

## Thinning
thin_interval <- 2
hmc.post_samples_phi_thin <- as.vector(window(hmc.post_samples_phi_mcmc, thin = thin_interval))
hmc.sigma_eta_thin <- as.vector(window(hmc.post_samples_sigma_eta_mcmc, thin = thin_interval))
hmc.sigma_eps_thin <- as.vector(window(hmc.post_samples_sigma_eps_mcmc, thin = thin_interval))


param_names <- c("phi", "sigma_eta", "sigma_eps")
param_dim <- length(param_names)
rvgaw.df <- data.frame(phi = rvgaw.post_samples_phi, 
                       sigma_eta = rvgaw.post_samples_eta, 
                       sigma_eps = rvgaw.post_samples_eps)
hmc.df <- data.frame(phi = hmc.post_samples_phi, 
                     sigma_eta = hmc.sigma_eta_thin, 
                     sigma_eps = hmc.sigma_eps_thin)
hmcw.df <- data.frame(phi = hmcw.post_samples_phi, 
                     sigma_eta = hmcw.post_samples_eta, 
                     sigma_eps = hmcw.post_samples_eps)
names(hmc.df) <- param_names
names(hmcw.df) <- param_names

## ggplot version
true_vals.df <- data.frame(phi = phi, sigma_eta = sigma_eta, sigma_eps = sigma_eps)
param_values <- c(phi, sigma_eta, sigma_eps)

plots <- list()

for (p in 1:param_dim) {
  
  true_vals.df <- data.frame(name = param_names[p], val = param_values[p])
  
  plot <- ggplot(rvgaw.df, aes(x=.data[[param_names[p]]])) +
    geom_density(col = "red", lwd = 1) +
    geom_density(data = hmcw.df, col = "goldenrod", lwd = 1) +
    geom_density(data = hmc.df, col = "deepskyblue", lwd = 1) +
    geom_vline(data = true_vals.df, aes(xintercept=val),
               color="black", linetype="dashed", linewidth=1) +
    labs(x = vars) +
    theme_bw() +
    theme(axis.title = element_blank(), text = element_text(size = 24)) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 4))
  
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
  
  param_df <- data.frame(x = param_values[q], y = param_values[p])
  
  cov_plot <- ggplot(rvgaw.df, aes(x = .data[[param_names[q]]], y = .data[[param_names[p]]])) +
    stat_ellipse(col = "red", type = "norm", lwd = 1) +
    stat_ellipse(data = hmcw.df, col = "goldenrod", type = "norm", lwd = 1) +
    stat_ellipse(data = hmc.df, col = "deepskyblue", type = "norm", lwd = 1) +
    geom_point(data = param_df, aes(x = x, y = y),
               shape = 4, color = "black", size = 4) +
    theme_bw() +
    theme(axis.title = element_blank(), text = element_text(size = 24)) +                               # Assign pretty axis ticks
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
gp <- gtable_add_rows(gp, unit(1.5, "lines"), -1) #0 adds on the top

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
  plot_file <- paste0("test_lgss_posterior", "_", n, temper_info, reorder_info, block_info,
                      "_", transform, "_", date, ".png")
  filepath = paste0("./plots/", plot_file)
  png(filepath, width = 800, height = 600)
  grid.draw(gp)
  dev.off()
}

## Trajectories
if (plot_trajectories) {
  mu_phi <- sapply(rvgaw_results$mu, function(x) x[1])
  mu_sigma_eta <- sapply(rvgaw_results$mu, function(x) x[2])
  mu_sigma_eps <- sapply(rvgaw_results$mu, function(x) x[3])
  if (transform == "arctanh") {
    mu_phi <- tanh(mu_phi)
  } else { # logit transform
    mu_phi <- exp(mu_phi) / (1 + exp(mu_phi))
  }
  mu_sigma_eta <- sqrt(exp(mu_sigma_eta))
  mu_sigma_eps <- sqrt(exp(mu_sigma_eps))
  plot_range <- 1:length(mu_phi) #400:1000#floor(n/2)

  true_df <- data.frame(param = c("phi", "sigma[eta]", "sigma[epsilon]"), 
                        value = c(phi, sigma_eta, sigma_eps))

  trajectory_df <- data.frame(phi = mu_phi, sigma_eta = mu_sigma_eta, sigma_eps = mu_sigma_eps)
  names(trajectory_df) <- c("phi", "sigma[eta]", "sigma[epsilon]")
  trajectory_df$iter <- 1:nrow(trajectory_df)

  trajectory_df_long <- trajectory_df %>% pivot_longer(cols = !iter, 
                                                      names_to = "param", values_to = "value")
  trajectory_plot <- trajectory_df_long %>% ggplot() + 
    geom_line(aes(x = iter, y = value), linewidth = 1) +
    facet_wrap(~param, scales = "free", labeller = label_parsed) +
    geom_hline(data = true_df, aes(yintercept = value), linetype = "dashed", linewidth = 1.5) +
    theme_bw() + theme(text = element_text(size = 30)) + 
    xlab("Iterations") + ylab("Value")
  print(trajectory_plot)                                                      

  png(paste0("./plots/trajectories_lgss", block_info, ".png"), width = 1500, height = 500)
  print(trajectory_plot)                                                      
  dev.off()
  # png("./plots/trajectory_lgss.png", width = 1500, height = 500)
  # par(mfrow = c(1,3))
  # plot(mu_phi[plot_range], type = "l",
  #      ylab = "phi", xlab = "Iterations", main = "Trajectory of phi")
  # abline(h = phi, lty = 2)
  
  # plot(mu_sigma_eta[plot_range], type = "l",
  #      ylab = "sigma_eta", xlab = "Iterations", main = "Trajectory of sigma_eta")
  # abline(h = sigma_eta, lty = 2)
  
  # plot(mu_sigma_eps[plot_range], type = "l",
  #      ylab = "sigma_eps", xlab = "Iterations", main = "Trajectory of sigma_eps")
  # abline(h = sigma_eps, lty = 2)
  # dev.off()
}

## Timing comparison
rvgaw.time <- rvgaw_results$time_elapsed[3]
hmcw.time <- sum(hmcw_results$time()$chains$total)
hmc.time <- sum(hmc_results$time()$chains$total)
print(data.frame(method = c("R-VGA", "HMCW", "HMC"),
                 time = c(rvgaw.time, hmcw.time, hmc.time)))
