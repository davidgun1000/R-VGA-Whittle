rm(list = ls())
setwd("~/R-VGA-Whittle/Multi_SV/")


source("./source/map_functions.R")
#######################
##   Generate data   ##
#######################

result_directory <- "./results/"

## Flags
date <- "20230814"
dataset <- "2"
regenerate_data <- F
plot_trajectory <- F

Tfin <- 10000
d <- 3

if (d == "2") {
  result_file <- ""
  model <- "bivariate_sv_"
} else {
  model <- "trivariate_sv_"
  result_file <- "trivariate_"
}

if (regenerate_data) {
  
  sigma_eta1 <- 0.9#4
  sigma_eta2 <- 1.5 #1.5
  sigma_eta3 <- 1.2 #1.5
  
  Sigma_eta <- diag(c(sigma_eta1, sigma_eta2, sigma_eta3))
  Sigma_eps <- diag(d)
  
  if (dataset == "0") {
    # Phi <- diag(c(0.7, 0.8, 0.9))
    Phi <- diag(c(0.9, 0.2, -0.8))
  } else if (dataset == "2") {
    Phi <- matrix(c(-0.3, -0.1,  0.4,
                    -0.2,  0.9,  0.8,
                    -0.2, -0.7, -0.8), 3, 3, byrow = T)
  } else if (dataset == "3") {
    Phi <- matrix(c(0.9, 0.0, -0.1,
                    -0.6, 0.2, 0.5,
                    1.0, 0.3, -0.8), 3, 3, byrow = T)
    # diag(Phi) <- c(0.7, 0.8, 0.9)
  } else { # generate a random Phi matrix using the mapping from unconstrained to constrained VAR(1) coef 
    A <- matrix(rnorm(d^2), d, d)
    Phi <- backward_map(A, Sigma_eta)
    Phi <- round(Phi, digits = 1)
  }
  
  set.seed(2022)
  x1 <- rmvnorm(1, rep(0, d), compute_autocov_VAR1(Phi, Sigma_eta))
  X <- matrix(NA, nrow = length(x1), ncol = Tfin+1) # x_0:T
  X[, 1] <- x1
  Y <- matrix(NA, nrow = length(x1), ncol = Tfin) # y_1:T
  # Y[, 1] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
  for (t in 1:Tfin) {
    X[, t+1] <- Phi %*% X[, t] + t(rmvnorm(1, rep(0, d), Sigma_eta))
    V <- diag(exp(X[, t+1]/2))
    Y[, t] <- V %*% t(rmvnorm(1, rep(0, d), Sigma_eps))
  }
  
  multi_sv_data <- list(X = X, Y = Y, Phi = Phi, Sigma_eta = Sigma_eta, 
                        Sigma_eps = Sigma_eps)
  
  if (save_data) {
    saveRDS(multi_sv_data, file = paste0("./data/", model, "data_Tfin", Tfin, "_", date, "_", dataset, ".rds"))
  }
} else {
  multi_sv_data <- readRDS(file = paste0("./data/", model, "data_Tfin", Tfin, "_", date, "_", dataset, ".rds"))
  
  X <- multi_sv_data$X
  Y <- multi_sv_data$Y
  Phi <- multi_sv_data$Phi
  Sigma_eta <- multi_sv_data$Sigma_eta
  Sigma_eps <- multi_sv_data$Sigma_eps
}

par(mfrow = c(d,1))

for (k in 1:d) {
  # plot(X[k, ], type = "l")
  plot(Y[k, ], type = "l")
}

############################## Inference #######################################

# ## Construct initial distribution/prior
# prior <- construct_prior(data = Y)
# prior_mean <- prior$prior_mean
# prior_var <- prior$prior_var
# 
# param_dim <- length(prior_mean)

################################
##    R-VGAW implementation   ##
################################

# if (use_tempering) {
#   n_temper <- 10
#   K <- 10
#   temper_schedule <- rep(1/K, K)
#   temper_info <- paste0("_temper", n_temper)
# } else {
#   temper_info <- ""
# }

# if (reorder_freq) {
#   reorder_info <- "_reorder"
# } else {
#   reorder_info <- ""
# }

S <- 100L
a_vals <- 1

################ R-VGA starts here #################

rvgaw_filepath_reorder <- paste0(result_directory, "rvga_whittle_results_", result_file, "Tfin", Tfin, 
                          "_temper10", "_reorder", "_", date, "_", dataset, ".rds")

rvgaw_filepath_noreorder <- paste0(result_directory, "rvga_whittle_results_", result_file, "Tfin", Tfin, 
                                 "_temper10", "_", date, "_", dataset, ".rds")

hmc_filepath <- paste0(result_directory, "hmc_results_", result_file, "Tfin", Tfin, 
                       "_", date, "_", dataset, ".rds")

rvgaw_results_reorder <- readRDS(rvgaw_filepath_reorder)
rvgaw_results_noreorder <- readRDS(rvgaw_filepath_noreorder)
stan_results <- readRDS(hmc_filepath)

rvgaw.post_samples_Phi_reorder <- rvgaw_results_reorder$post_samples$Phi
rvgaw.post_samples_Sigma_eta_reorder <- rvgaw_results_reorder$post_samples$Sigma_eta

rvgaw.post_samples_Phi_noreorder <- rvgaw_results_noreorder$post_samples$Phi
rvgaw.post_samples_Sigma_eta_noreorder <- rvgaw_results_noreorder$post_samples$Sigma_eta

hmc.post_samples_Phi <- stan_results$draws[,,1:(d^2)]
hmc.post_samples_Sigma_eta <- stan_results$draws[,,(d^2+1):(2*d^2)]

## Plot posterior estimates
indices <- data.frame(i = rep(1:d, each = d), j = rep(1:d, d))
hmc_indices <- c(matrix(1:d^2, d, d, byrow = T))
par(mfrow = c(d+1,d))

for (k in 1:nrow(indices)) {
  i <- indices[k, 1]
  j <- indices[k, 2]
  
  rvgaw.post_samples_phi_reorder <- unlist(lapply(rvgaw.post_samples_Phi_reorder, function(x) x[i,j]))
  rvgaw.post_samples_phi_noreorder <- unlist(lapply(rvgaw.post_samples_Phi_noreorder, function(x) x[i,j]))
  # mcmcw.post_samples_phi <- unlist(lapply(mcmcw.post_samples_Phi, function(x) x[i,j]))
  
  plot(density(rvgaw.post_samples_phi_reorder), col = "goldenrod",
       main = bquote(phi[.(c(i,j))]), xlim = Phi[i,j] + c(-0.3, 0.3))
  # lines(density(mcmcw.post_samples_phi), col = "blue", lty = 2)
  lines(density(rvgaw.post_samples_phi_noreorder), col = "red", lty = 2)
  lines(density(hmc.post_samples_Phi[,,hmc_indices[k]]), col = "forestgreen")
  abline(v = Phi[i,j], lty = 2)
  legend("topleft", legend = c("R-VGAW reorder", "R-VGAW no reorder", "HMC"), 
         col = c("goldenrod", "red", "forestgreen"),
         lty = c(1,2,1), cex = 0.4, y.intersp = 0.3)
}

# par(mfrow = c(1,d))
hmc_indices <- diag(matrix(1:(d^2), d, d))#c(1,5,9)
for (k in 1:d) {
  rvgaw.post_samples_sigma_eta_reorder <- unlist(lapply(rvgaw.post_samples_Sigma_eta_reorder, function(x) x[k,k]))
  rvgaw.post_samples_sigma_eta_noreorder <- unlist(lapply(rvgaw.post_samples_Sigma_eta_noreorder, function(x) x[k,k]))
  # mcmcw.post_samples_sigma_eta <- unlist(lapply(mcmcw.post_samples_Sigma_eta, function(x) x[k,k]))
  
  plot(density(rvgaw.post_samples_sigma_eta_reorder), col = "goldenrod",
       main = bquote(sigma_eta[.(c(k,k))]), xlim = Sigma_eta[k,k] + c(-0.4, 0.4))
  # lines(density(mcmcw.post_samples_sigma_eta), col = "blue", lty = 2)
  lines(density(rvgaw.post_samples_sigma_eta_noreorder), col = "red", lty = 2,)
  lines(density(hmc.post_samples_Sigma_eta[,,hmc_indices[k]]), col = "forestgreen")
  abline(v = Sigma_eta[k,k], lty = 2)
  legend("topleft", legend = c("R-VGAW reorder", "R-VGAW no reorder", "HMC"), 
         col = c("goldenrod", "red", "forestgreen"),
         lty = c(1,2,1), cex = 0.4, y.intersp = 0.3)
}

if (plot_trajectory) {
  ## Plot trajectories
  param_dim <- d^2 + d
  rvgaw.mu_trajec_reorder <- rvgaw_results_reorder$mu
  rvgaw.A_trajec_reorder <- lapply(rvgaw.mu_trajec_reorder, function(x) matrix(x[1:(d^2)], d, d, byrow = T))
  rvgaw.Sigma_trajec_reorder <- lapply(rvgaw.mu_trajec_reorder, function(x) diag(exp(x[(d^2+1):param_dim])))
  rvgaw.Phi_trajec_reorder <- mapply(backward_map, rvgaw.A_trajec_reorder, rvgaw.Sigma_trajec_reorder, SIMPLIFY = F)
  
  rvgaw.mu_trajec_noreorder <- rvgaw_results_noreorder$mu
  rvgaw.A_trajec_noreorder <- lapply(rvgaw.mu_trajec_noreorder, function(x) matrix(x[1:(d^2)], d, d, byrow = T))
  rvgaw.Sigma_trajec_noreorder <- lapply(rvgaw.mu_trajec_noreorder, function(x) diag(exp(x[(d^2+1):param_dim])))
  rvgaw.Phi_trajec_noreorder <- mapply(backward_map, rvgaw.A_trajec_noreorder, rvgaw.Sigma_trajec_noreorder, SIMPLIFY = F)
  
  
  for (k in 1:nrow(indices)) {
    i <- indices[k, 1]
    j <- indices[k, 2]
    
    rvgaw.phi_trajec_reorder <- unlist(lapply(rvgaw.Phi_trajec_reorder, function(x) x[i,j]))
    rvgaw.phi_trajec_noreorder <- unlist(lapply(rvgaw.Phi_trajec_noreorder, function(x) x[i,j]))
    
    plot(rvgaw.phi_trajec_reorder, type = "l", col = "goldenrod", 
         main = "yellow = reorder, red = no reorder")
    lines(rvgaw.phi_trajec_noreorder, col = "red", lty = 2)
    abline(h = Phi[i,j], lty = 2)
    
    # legend("bottomright", legend = c("R-VGAW reorder", "R-VGAW no reorder"), 
    #        col = c("goldenrod", "red"),
    #        lty = c(1,2,1), cex = 0.7, y.intersp = 0.5)
  }
  
  
  for (k in 1:nrow(indices)) {
    rvgaw.sigma_trajec_reorder <- unlist(lapply(rvgaw.Sigma_trajec_reorder, function(x) x[k,k]))
    rvgaw.sigma_trajec_noreorder <- unlist(lapply(rvgaw.Sigma_trajec_noreorder, function(x) x[k,k]))
    
    plot(rvgaw.sigma_trajec_reorder, type = "l", col = "goldenrod", 
         main = "yellow = reorder, red = no reorder")
    lines(rvgaw.sigma_trajec_noreorder, col = "red", lty = 2)
    abline(h = Sigma_eta[k,k], lty = 2)
    
    # legend("bottomright", legend = c("R-VGAW reorder", "R-VGAW no reorder"), 
    #        col = c("goldenrod", "red"),
    #        lty = c(1,2,1), cex = 0.7, y.intersp = 0.5)
  }
  
}