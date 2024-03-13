## Stochastic volatility model
setwd("~/R-VGA-Whittle/03_SV_real/")

rm(list = ls())

library(mvtnorm)
library(coda)
# library(Deriv)
# library(rstan)
library(cmdstanr)
library(tensorflow)
reticulate::use_condaenv("myenv", required = TRUE)
library(keras)
library(bspec)
library(ggplot2)
library(grid)
library(gridExtra)
library(gtable)

source("./source/compute_whittle_likelihood_sv.R")
# source("./source/run_rvgaw_sv_tf.R")
source("./source/run_rvgaw_sv_block.R")
source("./source/run_mcmc_sv.R")
source("./source/run_hmc_sv.R")
source("./source/compute_periodogram.R")
# source("./source/run_corr_pmmh_sv.R")
source("./source/particleFilter.R")

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

## Data
# date <- "20230223"
date <- "20230918"
dataset <- "exrates"

## R-VGA flags
use_tempering <- T
temper_first <- T
reorder <- 0 #"decreasing"
reorder_seed <- 2024
# n_reorder <- 10
plot_likelihood_surface <- F
prior_type <- ""
transform <- "arctanh"
plot_prior <- T
plot_trajectories <- F
use_welch <- F

## Flags
rerun_rvgaw <- T
rerun_mcmcw <- F
# rerun_mcmce <- F
rerun_hmc <- F
rerun_hmcw <- F

save_rvgaw_results <- F
save_mcmcw_results <- F
save_hmc_results <- F
save_hmcw_results <- F
save_plots <- F

n_post_samples <- 20000 # across chains
burn_in <- 10000 # across chains

nblocks <- 100
n_indiv <- 100

## Result directory
if (dataset == "sp100") {
  # result_directory <- paste0("./results/", prior_type, "/")
  result_directory <- paste0("./results/sp100/", transform, "/")
# } else if (dataset == "nasdaq"){
#   result_directory <- paste0("./results/nasdaq/", transform, "/")
} else if (dataset == "nasdaq") {
  result_directory <- paste0("./results/nasdaq/", transform, "/")
} else if (dataset == "exrates") {
  result_directory <- paste0("./results/exrates/", transform, "/")
} else {
  # result_directory <- paste0("./results/", prior_type, "/")
  result_directory <- paste0("./results/industry/", transform, "/")

}

## Read data
print("Reading saved data...")
y <- c()
# n <- 10000
if (dataset == "sp100") {
  sv_data <- read.csv(file = "./data/SP100.csv")
  nrows <- nrow(sv_data)
  ## Process the data to get daily returns
  sv_data$returns <- (sv_data$Close - sv_data$Open)/sv_data$Open
  sv_data$log_returns <- c(0, log(sv_data$Adj.Close[2:nrows]/sv_data$Adj.Close[1:(nrows-1)]) * 100)
  
  y <- sv_data$log_returns[-1]#[1:n]
  # y <- sv_data$returns
  # browser()
} else if (dataset == "nasdaq") {
  sv_data <- read.csv(file = "./data/nasdaq.csv")
  nrows <- nrow(sv_data)
  
  ## Process the data to get daily returns
  # sv_data$returns <- (sv_data$Close - sv_data$Open)/sv_data$Open
  sv_data$log_returns <- c(0, log(sv_data$Adj.Close[2:nrows]/sv_data$Adj.Close[1:(nrows-1)]) * 100)
  
  y <- sv_data$log_returns[-1]#[1:n]
} else if (dataset == "exrates") {
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
y <- exrates[, "NZD_returns"]
} else {
  sv_data <- read.csv(file = "./data/5_Industry_Portfolios_Daily_cleaned.csv")
  y <- sv_data[1:n, "Cnsmr"]
  # y <- sv_data[1:n, "HiTec"]
}

n <- length(y)

## Normalise data
y <- y - mean(y)

# par(mfrow = c(2,1))
plot(y, type = "l")

# plot(x, type = "l")

## Test likelihood computation
if (plot_likelihood_surface) {
  param_grid <- seq(0.01, 0.99, length.out = 100)
  llh1 <- c() # log likelihood when varying phi
  llh2 <- c() # log likelihood when varying sigma_eta
  
  for (k in 1:length(param_grid)) {
    params1 <- list(phi = param_grid[k], sigma_eta = sigma_eta, sigma_xi = sqrt(pi^2/2))
    params2 <- list(phi = phi, sigma_eta = param_grid[k], sigma_xi = sqrt(pi^2/2))
    
    llh1[k] <- compute_whittle_likelihood_sv(y = y, params = params1)
    llh2[k] <- compute_whittle_likelihood_sv(y = y, params = params2)
    
  }
  
  par(mfrow = c(1,2))
  plot_range <- 1:100
  plot(param_grid[plot_range], llh1[plot_range], type = "l", 
       xlab = "Parameter", ylab = "Log likelihood", main = "phi")
  abline(v = phi, lty = 2)
  abline(v = param_grid[which.max(llh1)], col = "red", lty = 2)
  
  plot_range <- 1:100
  plot(param_grid[plot_range], llh2[plot_range], type = "l", 
       xlab = "Parameter", ylab = "Log likelihood", main = "sigma_eta")
  abline(v = sigma_eta, lty = 2)
  abline(v = param_grid[which.max(llh2)], col = "red", lty = 2)
}


########################################
##                R-VGA               ##
########################################

S <- 1000L

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

if (!is.null(nblocks)) {
  block_info <- paste0("_", nblocks, "blocks", n_indiv, "indiv")
} else {
  block_info <- ""
}

## Prior
prior_info <- ""
if (prior_type == "prior1") {
  prior_mean <- c(0, -1) #rep(0,2)
  prior_var <- diag(c(1, 0.5)) #diag(1, 2)
  prior_info <- paste0("_", prior_type)
} else {
  prior_mean <- c(2, -3) #rep(0,2)
  # prior_mean <- c(0, -3) #rep(0,2)
  
  prior_var <- diag(c(0.5, 0.5)) #diag(1, 2)
}

if (plot_prior) {
  prior_theta <- rmvnorm(10000, prior_mean, prior_var)
  prior_phi <- c()
  if (transform == "arctanh") {
    prior_phi <- tanh(prior_theta[, 1])
  } else {
    prior_phi <- exp(prior_theta[, 1]) / (1 + exp(prior_theta[, 1]))
    
  }
  prior_eta <- sqrt(exp(prior_theta[, 2]))
  # prior_xi <- sqrt(exp(prior_theta[, 3]))
  # par(mfrow = c(2,1))
  hist(prior_phi, main = "Prior of phi") #, xlim = c(0.8, 1))
  hist(prior_eta, main = "Prior of sigma_eta")

}

# prior_theta <- rmvnorm(10000, prior_mean, prior_var)
# prior_phi <- tanh(prior_theta[, 1])
# prior_eta <- sqrt(exp(prior_theta[, 2]))
# prior_xi <- sqrt(exp(prior_theta[, 3]))
# par(mfrow = c(2,1))
# hist(prior_phi)
# hist(prior_eta)

rvgaw_filepath <- paste0(result_directory, "rvga_uni_real_results_n", n,
                         temper_info, reorder_info, block_info, prior_info, "_", date, ".rds")

if (rerun_rvgaw) {
  rvgaw_results <- run_rvgaw_sv(y = y, #sigma_eta = sigma_eta, sigma_eps = sigma_eps, 
                                prior_mean = prior_mean, prior_var = prior_var, 
                                n_post_samples = n_post_samples,
                                deriv = "tf", S = S, 
                                reorder = reorder,
                                reorder_seed = reorder_seed,
                                use_tempering = use_tempering, 
                                temper_first = temper_first,
                                n_temper = n_temper,
                                temper_schedule = temper_schedule, 
                                transform = transform,
                                # use_welch = use_welch,
                                nblocks = nblocks,
                                n_indiv = n_indiv)
  
  if (save_rvgaw_results) {
    saveRDS(rvgaw_results, rvgaw_filepath)
  }
  
} else {
  rvgaw_results <- readRDS(rvgaw_filepath)
}

rvgaw.post_samples_phi <- rvgaw_results$post_samples$phi
rvgaw.post_samples_sigma_eta <- rvgaw_results$post_samples$sigma_eta
# rvgaw.post_samples_xi <- rvgaw_results$post_samples$sigma_xi

########################################
##                MCMC                ## 
########################################

mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_results_n", n, prior_info,
                         "_", date, ".rds")

adapt_proposal <- T

# n_post_samples <- 10000
burn_in <- burn_in
MCMC_iters <- n_post_samples + burn_in

# prior_mean <- rep(0, 3)
# prior_var <- diag(c(1, 1, 0.01))

# prior_samples <- rmvnorm(10000, prior_mean, prior_var)
# prior_samples_phi <- tanh(prior_samples[, 1])                                       
# hist(prior_samples_sigma_xi)

if (rerun_mcmcw) {
  mcmcw_results <- run_mcmc_sv(y, #sigma_eta, sigma_eps, 
                               iters = MCMC_iters, burn_in = burn_in,
                               prior_mean = prior_mean, prior_var = prior_var,  
                               state_ini_mean = state_ini_mean, state_ini_var = state_ini_var,
                               adapt_proposal = T, use_whittle_likelihood = T,
                               transform = transform)
  
  if (save_mcmcw_results) {
    saveRDS(mcmcw_results, mcmcw_filepath)
  }
} else {
  mcmcw_results <- readRDS(mcmcw_filepath)
}

mcmcw.post_samples_phi <- as.mcmc(mcmcw_results$post_samples$phi[-(1:burn_in)])
mcmcw.post_samples_eta <- as.mcmc(mcmcw_results$post_samples$sigma_eta[-(1:burn_in)])
# mcmcw.post_samples_xi <- as.mcmc(mcmcw_results$post_samples$sigma_xi[-(1:burn_in)])

# par(mfrow = c(2,1))
# coda::traceplot(mcmcw.post_samples_phi, main = "Trace plot for phi")
# coda::traceplot(mcmcw.post_samples_eta, main = "Trace plot for sigma_eta")
# # traceplot(mcmcw.post_samples_xi, main = "Trace plot for sigma_xi")
# 
# par(mfrow = c(1,2))
# plot(density(mcmcw.post_samples_phi), main = "Posterior of phi", 
#      col = "blue", lty = 2, lwd = 2)
# lines(density(rvgaw.post_samples_phi), col = "red", lty = 2, lwd = 2)
# abline(v = phi, lty = 3)
# legend("topleft", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
#        col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)
# 
# plot(density(mcmcw.post_samples_eta), main = "Posterior of sigma_eta", 
#      col = "blue", lty = 2, lwd = 2)
# lines(density(rvgaw.post_samples_sigma_eta), col = "red", lty = 2, lwd = 2)
# abline(v = sigma_eta, lty = 3)
# legend("topright", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
#        col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)

# ####### MCMCE ##########
# mcmce_filepath <- paste0(result_directory, "mcmc_exact_results_n", n, 
#                          "_phi", phi_string, "_", date, ".rds")
# 
# if (rerun_mcmce) {
#   mcmce_results <- run_mcmc_sv(y, #sigma_eta, sigma_eps, 
#                                iters = MCMC_iters, burn_in = burn_in,
#                                prior_mean = prior_mean, prior_var = prior_var,  
#                                state_ini_mean = state_ini_mean, state_ini_var = state_ini_var,
#                                adapt_proposal = T, use_whittle_likelihood = F)
#   
#   if (save_mcmce_results) {
#     saveRDS(mcmce_results, mcmce_filepath)
#   }
# } else {
#   mcmce_results <- readRDS(mcmce_filepath)
# }
# 
# mcmce.post_samples_phi <- as.mcmc(mcmce_results$post_samples$phi[-(1:burn_in)])
# mcmce.post_samples_eta <- as.mcmc(mcmce_results$post_samples$sigma_eta[-(1:burn_in)])
# # mcmce.post_samples_xi <- as.mcmc(mcmce_results$post_samples$sigma_xi[-(1:burn_in)])
# 
# par(mfrow = c(2,1))
# coda::traceplot(mcmce.post_samples_phi, main = "Trace plot for phi")
# coda::traceplot(mcmce.post_samples_eta, main = "Traceplot for sigma_eta")
# # traceplot(mcmce.post_samples_xi, main = "Trace plot for sigma_xi")

### STAN ###
hmc_filepath <- paste0(result_directory, "hmc_results_n", n, prior_info,
                        "_", date, ".rds")

n_chains <- 2

if (rerun_hmc) {
  hmc_results <- run_hmc_sv(data = y, transform = transform,
                             iters = n_post_samples / n_chains, 
                             burn_in = burn_in / n_chains,
                             n_chains = n_chains,
                             prior_mean = prior_mean, prior_var = prior_var)
  
  if (save_hmc_results) {
    saveRDS(hmc_results, hmc_filepath)
  }
  
} else {
  hmc_results <- readRDS(hmc_filepath)
}

hmc.post_samples_phi <- c(hmc_results$draws[,,1]) #tanh(hmc.theta_phi)
hmc.post_samples_sigma_eta <- c(hmc_results$draws[,,2])#sqrt(exp(hmc.theta_sigma))

########################################################
##          Stan with the Whittle likelihood          ##
########################################################
hmcw_filepath <- paste0(result_directory, "hmcw_results_n", n, prior_type,
                        "_", date, ".rds")

if (rerun_hmcw) {
  # n_chains <- 2
  # hmc_iters <- n_post_samples / n_chains
  # burn_in <- 5000 # per chain

  # Compute periodogram
  pgram_out <- compute_periodogram(y)
  freq <- pgram_out$freq
  I <- pgram_out$periodogram

  # ## Fourier frequencies
  # k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
  # k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
  # freq <- 2 * pi * k_in_likelihood / n

  # ## Fourier transform of the observations
  # y_tilde <- log(y^2) - mean(log(y^2))

  # fourier_transf <- fft(y_tilde)
  # periodogram <- 1/n * Mod(fourier_transf)^2
  # I <- periodogram[k_in_likelihood + 1]

  whittle_stan_file <- "./source/stan_sv_whittle.stan"
  # whittle_stan_file <- "./source/stan_mwe.stan" # this was to test the use of complex numbers

  whittle_sv_model <- cmdstan_model(
    whittle_stan_file,
    cpp_options = list(stan_threads = TRUE)
  )

  # log_kappa2_est <- mean(log(y^2)) - (digamma(1/2) + log(2))

  whittle_sv_data <- list(nfreq = length(freq), freqs = freq, periodogram = I,
                          prior_mean = prior_mean, diag_prior_var = diag(prior_var),
                          transform = ifelse(transform == "arctanh", 1, 0))

  # hfit <- stan(model_code = sv_code, 
  #              model_name="sv", data = sv_data, 
  #              iter = iters, warmup = burn_in, chains=1)

  fit_stan_multi_sv_whittle <- whittle_sv_model$sample(
    whittle_sv_data,
    chains = n_chains,
    threads = parallel::detectCores(),
    refresh = 100,
    iter_warmup = burn_in / n_chains,
    iter_sampling = n_post_samples / n_chains
  )

  hmcw_results <- list(draws = fit_stan_multi_sv_whittle$draws(variables = c("phi", "sigma_eta")),
                      time = fit_stan_multi_sv_whittle$time)

  # hmcw.fit <- extract(fit_stan_multi_sv_whittle, pars = c("theta_phi", "theta_sigma"),
  #                    permuted = F, inc_warmup = T)
  
  if (save_hmcw_results) {
    saveRDS(hmcw_results, hmcw_filepath)
  }

} else {
  hmcw_results <- readRDS(hmcw_filepath)
}

hmcw.post_samples_phi <- c(hmcw_results$draws[,,1])
hmcw.post_samples_sigma_eta <- c(hmcw_results$draws[,,2])


###########################

par(mfrow = c(1,2))
plot(density(rvgaw.post_samples_phi), main = "Posterior of phi", 
     col = "red", lty = 2, lwd = 3) #, xlim = c(0.9, 0.999))
# lines(density(mcmce.post_samples_phi), col = "blue", lwd = 3)
lines(density(mcmcw.post_samples_phi), col = "goldenrod", lty = 2, lwd = 3)
# lines(density(hmc.post_samples_phi), col = "deepskyblue", lty = 1, lwd = 3)
lines(density(hmcw.post_samples_phi), col = "royalblue", lty = 2, lwd = 3)
# legend("topleft", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
      #  col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)

plot(density(rvgaw.post_samples_sigma_eta), main = "Posterior of sigma_eta", 
     col = "red", lty = 2, lwd = 3) #, xlim = c(0.05, 0.5))
# lines(density(mcmce.post_samples_eta), col = "blue", lwd = 3)
lines(density(mcmcw.post_samples_eta), col = "goldenrod", lty = 2, lwd = 3)
# lines(density(hmc.post_samples_sigma_eta), col = "deepskyblue", lty = 1, lwd = 3)
lines(density(hmcw.post_samples_sigma_eta), col = "royalblue", lty = 2, lwd = 3)
# legend("topright", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
      #  col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)


## Trajectories
if (plot_trajectories) {
  mu_phi <- sapply(rvgaw_results$mu, function(x) x[1])
  mu_eta <- sapply(rvgaw_results$mu, function(x) x[2])
  
  par(mfrow = c(1, 2))
  if (transform == "arctanh") {
    plot(tanh(mu_phi), type = "l", main = "Trajectory of phi")
  } else {
    plot(1 / (1 + exp(-mu_phi)), type = "l", main = "Trajectory of phi")
  }
  
  plot(sqrt(exp(mu_eta)), type = "l", main = "Trajectory of sigma_eta")
}

## Estimation of kappa
mean_log_eps2 <- digamma(1/2) + log(2)
log_kappa2 <- mean(log(y^2)) - mean_log_eps2
kappa <- sqrt(exp(log_kappa2))


# if (save_plots) {
#   plot_file <- paste0("sv_real_posterior_", n, temper_info, reorder_info,
#                       "_", transform, "_", date, ".png")
#   filepath = paste0("./plots/", plot_file)
#   png(filepath, width = 600, height = 350)
#   
#   par(mfrow = c(1,2))
#   plot(density(rvgaw.post_samples_phi), main = "Posterior of phi", 
#       col = "red", lty = 2, lwd = 3) #, xlim = c(0.9, 0.999))
#   # lines(density(mcmce.post_samples_phi), col = "blue", lwd = 3)
#   lines(density(mcmcw.post_samples_phi), col = "goldenrod", lty = 2, lwd = 3)
#   # lines(density(hmc.post_samples_phi), col = "deepskyblue", lty = 1, lwd = 3)
#   lines(density(hmcw.post_samples_phi), col = "royalblue", lty = 2, lwd = 3)
#   # legend("topleft", legend = c("MCMC exact", "MCMC Whittle", "R-VGA Whittle"),
#         #  col = c("blue", "blue", "red"), lty = c(1, 2, 2), cex = 0.7)
# 
#   plot(density(rvgaw.post_samples_sigma_eta), main = "Posterior of sigma_eta", 
#       col = "red", lty = 2, lwd = 3) #, xlim = c(0.05, 0.5))
#   # lines(density(mcmce.post_samples_eta), col = "blue", lwd = 3)
#   lines(density(mcmcw.post_samples_eta), col = "goldenrod", lty = 2, lwd = 3)
#   # lines(density(hmc.post_samples_sigma_eta), col = "deepskyblue", lty = 1, lwd = 3)
#   lines(density(hmcw.post_samples_sigma_eta), col = "royalblue", lty = 2, lwd = 3)
#   
#   dev.off()
# }

## ggplot version
param_names <- c("phi", "sigma_eta")
param_dim <- length(param_names)
rvgaw.df <- data.frame(phi = rvgaw.post_samples_phi, 
                       sigma_eta = rvgaw.post_samples_sigma_eta)
hmc.df <- data.frame(phi = hmc.post_samples_phi, 
                     sigma_eta = hmc.post_samples_sigma_eta)
hmcw.df <- data.frame(phi = hmcw.post_samples_phi, 
                      sigma_eta = hmcw.post_samples_sigma_eta)
names(hmc.df) <- param_names
names(hmcw.df) <- param_names

## ggplot version

plots <- list()

for (p in 1:param_dim) {
  
  plot <- ggplot(rvgaw.df, aes(x=.data[[param_names[p]]])) +
    # plot <- ggplot(exact_rvgal.df, aes(x=colnames(exact_rvgal.df)[p])) + 
    geom_density(col = "red", lwd = 1) +
    geom_density(data = hmcw.df, col = "goldenrod", lwd = 1) +
    geom_density(data = hmc.df, col = "deepskyblue", lwd = 1) +
    labs(x = vars) +
    theme_bw() +
    theme(axis.title = element_blank(), text = element_text(size = 24)) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 4))
  # theme(legend.position="bottom") + 
  # scale_color_manual(values = c('RVGA' = 'red', 'HMC' = 'blue'))
  
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
  
  cov_plot <- ggplot(rvgaw.df, aes(x = .data[[param_names[q]]], y = .data[[param_names[p]]])) +
    stat_ellipse(col = "red", type = "norm", lwd = 1) +
    stat_ellipse(data = hmcw.df, col = "goldenrod", type = "norm", lwd = 1) +
    stat_ellipse(data = hmc.df, col = "deepskyblue", type = "norm", lwd = 1) +
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

gtable_show_layout(gp)

# Add the label grobs.
# The labels on the left should be rotated; hence the edit.
# t and l refer to cells in the gtable layout.
# gtable_show_layout(gp) shows the layout.
gp <- gtable_add_grob(gp, lapply(vars[1:param_dim], editGrob, rot = 90), t = 1:param_dim, l = 1)
gp <- gtable_add_grob(gp, vars[1:param_dim], t = param_dim+1, l = 2:(param_dim+1))

grid.newpage()
grid.draw(gp)

if (save_plots) {
  plot_file <- paste0("sv_real_posterior", "_", n, temper_info, reorder_info,
                      "_", transform, "_", date, ".png")
  filepath = paste0("./plots/", plot_file)
  png(filepath, width = 800, height = 600)
  grid.draw(gp)
  dev.off()
}

rvgaw.time <- rvgaw_results$time_elapsed[3]
hmcw.time <- sum(hmcw_results$time()$chains$total)
hmc.time <- sum(hmc_results$time()$chains$total)
print(data.frame(method = c("R-VGA", "HMCW", "HMC"),
                 time = c(rvgaw.time, hmcw.time, hmc.time)))
