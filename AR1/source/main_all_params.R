setwd("~/R-VGA-Whittle/AR1")

## R-VGA with Whittle likelihood (R-VGAW)?
rm(list = ls())

# library(stats)
# library(LSTS)
library(Matrix)
library(Deriv)
library(coda)
library(ggplot2)
library(tidyr)
library(mvtnorm)
reticulate::use_condaenv("tf2.11", required = TRUE)
library(tensorflow)
library(keras)

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

source("./source/calculate_likelihood.R")
source("./source/run_rvgaw_ar1_tf.R")
source("./source/run_rvgae_ar1_dev.R")
source("./source/run_mcmc_ar1_allparams.R")
source("./source/run_vb_ar1.R")

result_directory <- "./results/"

## Flags
date <- "20230417" # 20240410 has phi = 0.9, 20230417 has phi = 0.7
regenerate_data <- F
rerun_rvgaw <- T
rerun_rvgae <- T
# rerun_vbw <- F
# rerun_vbe <- F
rerun_mcmc_whittle <- T
rerun_mcmc_exact <- T
adapt_proposal <- T
use_matlab_deriv <- T

save_data <- F
save_rvgaw_results <- T
save_rvgae_results <- T
# save_vbw_results <- F
# save_vbe_results <- F
save_mcmcw_results <- T
save_mcmce_results <- T
save_plots <- F

## R-VGA flags
use_tempering <- T
reorder_freq <- T
reorder_seed <- 2024
decreasing <- F
transform <- "arctanh"

## MCMC flags
n_post_samples <- 10000
burn_in <- 5000
MCMC_iters <- n_post_samples + burn_in

## Model parameters 
phi <- 0.9
sigma_e <- 0.5
n <- 10000 # time series length

if (use_tempering) {
  n_temper <- 10 #0.1 * n #floor(n/2) #10
  K <- 10
  temper_schedule <- rep(1/K, K)
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
  set.seed(2023)
  x0 <- 1 #rnorm(1, 0, 1) 
  x <- c()
  x[1] <- x0
  
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

plot(1:n, x, type = "l")

####################################################
##         R-VGA with Whittle likelihood          ##
####################################################

rvgaw_filepath <- paste0(result_directory, "rvga_whittle_results_allparams_n", n, 
                         "_phi", phi_string, reorder_info, "_", date, ".rds")

S <- 200L

# Priors: theta_phi ~ N(0, 1), theta_sigma ~ N(0, 0.5)
# 
# mu_0 <- c(log(phi/(1-phi)), log(sigma_e^2))
if (transform == "arctanh") {
  # mu_0 <- c(atanh(phi), log(sigma_e^2))
  mu_0 <- c(0, 0)
  P_0 <- diag(c(1, 1))
} else {
  mu_0 <- c(0, 0) #c(2, -1)
  P_0 <- diag(c(2, 1)) #diag(c(0.5, 1))
}

# mu_0 <- 0
# P_0 <- 1

par(mfrow = c(2,1))
test_theta_phi <- rnorm(10000, mu_0[1], P_0[1,1])
if (transform == "arctanh") {
  phi_test <- tanh(test_theta_phi)
} else {
  phi_test <- exp(test_theta_phi) / (1 + exp(test_theta_phi))
}
hist(phi_test)
test_theta_sigma <- rnorm(10000, mu_0[2], P_0[2,2])
sigma_e_test <- sqrt(exp(test_theta_sigma))
hist(sigma_e_test)

browser()

if (rerun_rvgaw) {
  
  rvgaw_results <- run_rvgaw_ar1(series = x, #sigma_e = sigma_e, 
                                 prior_mean = mu_0, prior_var = P_0, 
                                 S = S, deriv = "", 
                                 use_tempering = use_tempering,
                                 reorder_freq = reorder_freq,
                                 decreasing = decreasing, 
                                 reorder_seed = reorder_seed,
                                 transform = transform)
  
  if (save_rvgaw_results) {
    saveRDS(rvgaw_results, rvgaw_filepath)
  }
  
} else {
  rvgaw_results <- readRDS(rvgaw_filepath)
}
rvgaw.post_samples <- rvgaw_results$post_samples

###############################################
##        R-VGA with exact likelihood        ##
###############################################
rvgae_filepath <- paste0(result_directory, "rvga_exact_results_allparams_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

if (rerun_rvgae) {
  
  rvgae_results <- run_rvgae_ar1(series = x, sigma_e = sigma_e, 
                                 prior_mean = mu_0, prior_var = P_0, 
                                 S = S, deriv = "deriv", 
                                 use_tempering = use_tempering,
                                 use_matlab_deriv = T)
  ## Save results
  if (save_rvgae_results) {
    saveRDS(rvgae_results, rvgae_filepath)
  }
  
} else {
  rvgae_results <- readRDS(rvgae_filepath)
}
rvgae.post_samples <- rvgae_results$post_samples

if (length(mu_0) > 1) {
  par(mfrow = c(2,1))
  plot(density(rvgaw.post_samples$phi), col = "red", lty = 2, 
       main = "Posterior density of phi")
  lines(density(rvgae.post_samples$phi), col = "red")
  abline(v = phi, lty = 2)
  legend("topleft", legend = c("R-VGA Whittle", "R-VGA exact"),
         col = c("red", "red"), lty = c(2,1), cex = 0.5)
  
  plot(density(rvgaw.post_samples$sigma), col = "red", lty = 2,
       main = "Posterior density of sigma")
  lines(density(rvgae.post_samples$sigma), col = "red")
  abline(v = sigma_e, lty = 2)
  legend("topleft", legend = c("R-VGA Whittle", "R-VGA exact"),
         col = c("red", "red"), lty = c(2,1), cex = 0.5)
}

# ##############################################
# ##      Batch VB with exact likelihood      ##
# ##############################################
# vbe_filepath <- paste0(result_directory, "vb_exact_results_n", n, 
#                        "_phi", phi_string, "_", date, ".rds")
# 
# if (rerun_vbe) {
#   print("Starting batch VB with exact likelihood...")
#   vbe_results <- run_vb_ar1(series = x, mu_0 = mu_0, sigma_0 = P_0, 
#                             sigma_e = sigma_e, 
#                             VB_iters = 2000, use_whittle_likelihood = F)
#   
#   if (save_vbe_results) {
#     saveRDS(vbe_results, vbe_filepath)
#   }
#   
# } else {
#   vbe_results <- readRDS(vbe_filepath)
# }
# # vbe.post_samples <- rnorm(n_post_samples, 
# #                           vbe_results$mu[[length(vbe_results$mu)]],
# #                           sqrt(vbe_results$sigma[[length(vbe_results$sigma)]]))
# 
# vbe.post_samples <- vbe_results$post_samples
# 
# ##############################################
# ##     Batch VB with Whittle likelihood     ##
# ##############################################
# vbw_filepath <- paste0(result_directory, "vb_whittle_results_n", n, 
#                        "_phi", phi_string, "_", date, ".rds")
# 
# if (rerun_vbw) {
#   print("Starting batch VB with Whittle likelihood...")
#   vbw_results <- run_vb_ar1(series = x, mu_0 = mu_0, sigma_0 = P_0, 
#                             sigma_e = sigma_e, 
#                             VB_iters = 2000, use_whittle_likelihood = T)
#   
#   if (save_vbw_results) {
#     saveRDS(vbw_results, vbw_filepath)
#   }
#   
# } else {
#   vbw_results <- readRDS(vbw_filepath)
# }
# # vbw.post_samples <- rnorm(n_post_samples, 
# #                           vbw_results$mu[[length(vbw_results$mu)]],
# #                           sqrt(vbw_results$sigma[[length(vbw_results$sigma)]]))
# 
# vbw.post_samples <- vbw_results$post_samples
# 
# ##############################################
##        MCMC with Whittle likelihood      ##
##############################################

mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_results_allparams_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

if (rerun_mcmc_whittle) {
  print("Starting MCMC with Whittle likelihood...")
  mcmcw_results <- run_mcmc_ar1(series = x, #sigma_e =  sigma_e, 
                                iters = MCMC_iters, burn_in = burn_in, 
                                prior_mean = mu_0, prior_var = P_0, 
                                adapt_proposal = T, use_whittle_likelihood = T)
  
  if (save_mcmcw_results) {
    saveRDS(mcmcw_results, mcmcw_filepath)
  }
  
} else {
  mcmcw_results <- readRDS(mcmcw_filepath)
}

mcmcw.post_samples <- as.mcmc(mcmcw_results$post_samples[-(1:burn_in), ])

mcmcw.post_samples_phi <- mcmcw.post_samples[, 1]
mcmcw.post_samples_sigma <- mcmcw.post_samples[, 2]

##############################################
##        MCMC with exact likelihood        ##
##############################################

mcmce_filepath <- paste0(result_directory, "mcmc_exact_results_allparams_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

if (rerun_mcmc_exact) {
  print("Starting MCMC with exact likelihood...")
  mcmce_results <- run_mcmc_ar1(series = x, #sigma_e =  sigma_e, 
                                iters = MCMC_iters, burn_in = burn_in,
                                prior_mean = mu_0, prior_var = P_0, 
                                adapt_proposal = T, use_whittle_likelihood = F)
  if (save_mcmce_results) {
    saveRDS(mcmce_results, mcmce_filepath)
  }
} else {
  mcmce_results <- readRDS(mcmce_filepath)
}

mcmce.post_samples <- as.mcmc(mcmce_results$post_samples[-(1:burn_in), ])
mcmce.post_samples_phi <- mcmce.post_samples[, 1]
mcmce.post_samples_sigma <- mcmce.post_samples[, 2]

par(mfrow = c(1, 2))
plot(density(mcmce.post_samples_phi), xlab = "phi", xlim = c(phi - 0.03, phi + 0.005), 
     col = "blue", main = paste0("Posterior of phi"), lwd = 2)
lines(density(mcmcw.post_samples_phi), col = "blue", lty = 3, lwd = 2)
lines(density(rvgae.post_samples$phi), col = "red", lwd = 2)
lines(density(rvgaw.post_samples$phi), col = "red", lty = 3, lwd = 2)
abline(v = phi, col = "black", lty = 2, lwd = 2)
legend("topleft", legend = c("MCMC exact", "MCMC Whittle", "R-VGA exact", "R-VGA Whittle"),
       col = c("blue", "blue", "red", "red"), lty = c(1,3,1,3), lwd = 2, cex = 0.5)

plot(density(mcmce.post_samples_sigma), xlab = "sigma_eta", #xlim = c(0.88, 0.92), 
     col = "blue", main = paste0("Posterior of sigma"), lwd = 2)
lines(density(mcmcw.post_samples_sigma), col = "blue", lty = 3, lwd = 2)
lines(density(rvgae.post_samples$sigma), col = "red", lwd = 2)
lines(density(rvgaw.post_samples$sigma), col = "red", lty = 3, lwd = 2)
abline(v = sigma_e, col = "black", lty = 2, lwd = 2)
legend("topleft", legend = c("MCMC exact", "MCMC Whittle", "R-VGA exact", "R-VGA Whittle"),
       col = c("blue", "blue", "red", "red"), lty = c(1,3,1,3), lwd = 2, cex = 0.5)


if (save_plots) {
  png(paste0("./plots/allparams_results_n", n, "_phi", phi_string,
             temper_info, reorder_info, ".png"), width = 600, height = 450)
  plot(density(mcmce.post_samples_phi), xlab = "phi", #xlim = c(0.88, 0.92), 
       col = "blue", main = paste0("Posterior of phi"), lwd = 2)
  lines(density(mcmcw.post_samples_phi), col = "blue", lty = 3, lwd = 2)
  lines(density(rvgae.post_samples$phi), col = "red", lwd = 2)
  lines(density(rvgaw.post_samples$phi), col = "red", lty = 3, lwd = 2)
  abline(v = phi, col = "black", lty = 2, lwd = 2)
  legend("topleft", legend = c("MCMC exact", "MCMC Whittle", "R-VGA exact", "R-VGA Whittle"),
         col = c("blue", "blue", "red", "red"), lty = c(1,3,1,3), lwd = 2, cex = 0.5)
  
  plot(density(mcmce.post_samples_sigma), xlab = "sigma_eta", #xlim = c(0.88, 0.92), 
       col = "blue", main = paste0("Posterior of sigma"), lwd = 2)
  lines(density(mcmcw.post_samples_sigma), col = "blue", lty = 3, lwd = 2)
  lines(density(rvgae.post_samples$sigma), col = "red", lwd = 2)
  lines(density(rvgaw.post_samples$sigma), col = "red", lty = 3, lwd = 2)
  abline(v = sigma_e, col = "black", lty = 2, lwd = 2)
  legend("topleft", legend = c("MCMC exact", "MCMC Whittle", "R-VGA exact", "R-VGA Whittle"),
         col = c("blue", "blue", "red", "red"), lty = c(1,3,1,3), lwd = 2, cex = 0.5)
  
  dev.off()
}


# Trace plot
# traceplot(as.mcmc(mcmce.post_samples), main = "Trace plot")

## Compare with exact likelihood
# mcmce_results <- readRDS(file = paste0(result_directory, "mcmc_gaussian_results_n", n, "_", date, ".rds"))
# gaussian_mcmc_post_samples <- gaussian_mcmc_results$post_samples[-(1:burn_in)]

# margin <- 0.02
# 
# plot(density(mcmce.post_samples), xlab = "phi", xlim = c(phi - margin, phi + margin),
#      col = "blue", main = paste0("Posterior draws (series length = ", n, ")"),
#      lwd = 2)
# lines(density(mcmcw.post_samples), col = "blue", lty = 3, lwd = 2)
# lines(density(rvgae.post_samples), col = "red", lwd = 2)
# lines(density(rvgaw.post_samples), col = "red", lty = 3, lwd = 2)
# lines(density(vbe.post_samples), col = "goldenrod", lwd = 2)
# lines(density(vbw.post_samples), col = "goldenrod", lty = 3, lwd = 2)
# abline(v = phi, col = "black", lty = 2, lwd = 2)
# legend("topright", legend = c("MCMC exact", "MCMC Whittle",
#                               "R-VGA exact",  "R-VGA Whittle",
#                               "Batch VB exact",  "Batch VB Whittle"),
#        col = c("blue", "blue", "red", "red", "goldenrod", "goldenrod"),
#        lty = c(1, 3, 1, 3, 1, 3), lwd = 2, cex = 0.8)
# 
# if (save_plots) {
#   png(paste0("./plots/rvga_mcmc_whittle_results_n", n, "_phi", phi_string, 
#              temper_info, reorder_info, ".png"), width = 600, height = 450)
#   plot(density(mcmce.post_samples), xlab = "phi", xlim = c(phi - margin, phi + margin),
#        col = "blue", main = paste0("Posterior draws (series length = ", n, ")"),
#        lwd = 2)
#   lines(density(mcmcw.post_samples), col = "blue", lty = 2, lwd = 2)
#   lines(density(rvgae.post_samples), col = "red", lwd = 2)
#   lines(density(rvgaw.post_samples), col = "red", lty = 2, lwd = 2)
#   lines(density(vbe.post_samples), col = "goldenrod", lwd = 2)
#   lines(density(vbw.post_samples), col = "goldenrod", lty = 2, lwd = 2)
#   abline(v = phi, col = "black", lty = 2, lwd = 2)
#   legend("topright", legend = c("MCMC exact", "MCMC Whittle",
#                                 "R-VGA exact",  "R-VGA Whittle",
#                                 "Batch VB exact",  "Batch VB Whittle"),
#          col = c("blue", "blue", "red", "red", "goldenrod", "goldenrod"),
#          lty = c(1, 2, 1, 2, 1, 2), lwd = 2)
# 
#   dev.off()
# }

# # ## ggplot version
# all_post_samples <- c(rvgaw.post_samples, rvgae.post_samples,
#                       # vbw.post_samples, vbe.post_samples,
#                       mcmcw.post_samples, mcmce.post_samples)
# 
# method_names <- rep(c("rvgaw", "rvgae", 
#                       # "vbw", "vbe", 
#                       "mcmcw", "mcmce"), each = n_post_samples)
# likelihood_type <- rep(c(rep("Whittle", n_post_samples), rep("Exact", n_post_samples)), times = 2)
# 
# # all_post_samples <- c(rvgae.post_samples,
# #                       vbe.post_samples,
# #                       mcmce.post_samples)
# # 
# # method_names <- rep(c("rvgae", "vbe", "mcmce"), each = n_post_samples)
# # likelihood_type <- rep(rep("Exact", n_post_samples), times = 3)
# 
# 
# all_df <- data.frame(method = method_names, likelihood = likelihood_type,
#                      post_samples = all_post_samples)
# true_vals.df <- data.frame(phi = phi)
# 
# g <- ggplot(all_df, aes(x = post_samples, colour = method, linetype = likelihood)) +
#   geom_density(linewidth = 1) +
#   geom_vline(data = true_vals.df, aes(xintercept=phi),
#              color="black", linetype="dashed", linewidth=0.75) +
#   # guides(color=guide_legend(title="Condition")) +
#   scale_color_manual(name = "method", values=c("blue", "blue", #"salmon",
#                                                "red", "red")) #, #"#56B4E9",
#                                                # "goldenrod", "goldenrod")) + #, "gold")) +
#   scale_linetype_manual(name = "method", values=rep(c(1, 2), 2)) +
#   theme_bw() + 
#   theme(axis.title = element_blank(), text = element_text(size = 24)) #+                               # Assign pretty axis ticks
# # scale_x_continuous(breaks = scales::pretty_breaks(n = 4)) 
# print(g)
# 
# if (save_plots) {
#   png(paste0("./plots/rvga_mcmc_whittle_results_n", n, "_phi", phi_string, 
#              temper_info, reorder_info, ".png"), width = 800, height = 500)
#   print(g)
#   dev.off()
# }
# 
# #######################################
# #      MCMC: INSPECT EFFICIENCY      ##
# #######################################
# # Get autocorrelation from CODA
# autocorr(mcmcw.post_samples, lags = c(0, 1, 5, 10, 50), relative=TRUE)
# lag.max <- 20
# 
# # ask <- FALSE;
# # autocorr.plot(draws, lag.max, auto.layout = TRUE, ask)
# # dev.off()
# 
# # Effective sample size and inefficiency factors
# # Compute effective sample size (ESS). This is of course done after burn-in
# ESS <- effectiveSize(mcmcw.post_samples)
# cat("MCMC ESS =", ESS)
# 
# # Compute Inefficiency factor
# # IF <- dim(draws)[1]/ESS
# # print(IF)
# 
# ## Trajectories
# plot_range <- 1:10 #400:1000#floor(n/2)
# if (reorder_freq) {
#   plot_title <- c("Trajectories with randomly reordered frequencies")
# } else {
#   plot_title <- c("Trajectories with original order of frequencies")
# }
# 
# if (transform == "arctanh") {
#   # rvgae_mu <- tanh(unlist(rvgae_results$mu)[plot_range])
#   rvgaw_mu <- tanh(unlist(rvgaw_results$mu)[plot_range])
# } else {
#   # rvgae_mu <- unlist(rvgae_results$mu)[plot_range]
#   # rvgae_mu <- exp(rvgae_mu) / (1 + exp(rvgae_mu))
#   rvgaw_mu <- unlist(rvgaw_results$mu)[plot_range]
#   rvgaw_mu <- exp(rvgaw_mu) / (1 + exp(rvgaw_mu))
# }
# 
# # par(mfrow = c(1,1))
# plot(tanh(unlist(rvgae_results$mu)[plot_range]), type = "l", ylim = c(0, 1),
#      ylab = "mu", xlab = "Iterations", main = plot_title)
# lines(rvgaw_mu[plot_range], col = "red")
# abline(h = phi, lty = 2)
# legend("bottomright", legend = c("R-VGA Exact", "R-VGA Whittle"), col = c("black", "red"),
#        lty = 1)
