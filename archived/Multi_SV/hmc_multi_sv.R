## Bivariate SV model

setwd("~/R-VGA-Whittle/Multi_SV/")
rm(list = ls())

#library("rstan")
library(cmdstanr)
source("./source/compute_whittle_likelihood_sv.R")
source("./source/compute_whittle_likelihood_multi_sv.R")
source("./source/map_functions.R")
source("./source/construct_prior.R")

library("mvtnorm")
library("astsa")
# library("expm")
library("stcos")

date <- "20230814"
result_directory <- "./results/"

## Flags
regenerate_data <- T
save_data <- F

rerun_hmc <- T
save_hmc_results <- F

dataset <- "1"

Tfin <- 1000
if (regenerate_data) {
  phi11 <- 0.7
  phi12 <- -0.5
  phi21 <- 0.2
  phi22 <- 0.9
  
  if (dataset == "0") {
    Phi <- diag(c(phi11, phi22))  
  } else {
    Phi <- matrix(c(phi11, phi12, phi21, phi22), 2, 2, byrow = T)
  }
  
  sigma_eta1 <- 0.9#4
  sigma_eta2 <- 1.5#7
  Sigma_eta <- diag(c(sigma_eta1, sigma_eta2))
  
  sigma_eps1 <- 1 #0.01
  sigma_eps2 <- 1 #0.02
  
  Sigma_eps <- diag(c(sigma_eps1, sigma_eps2))
  
  x1 <- c(0, 0)
  X <- matrix(NA, nrow = length(x1), ncol = Tfin+1) # x_0:T
  X[, 1] <- x1
  Y <- matrix(NA, nrow = length(x1), ncol = Tfin) # y_1:T
  # Y[, 1] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
  set.seed(2022)
  for (t in 1:Tfin) {
    X[, t+1] <- Phi %*% X[, t] + t(rmvnorm(1, c(0, 0), Sigma_eta))
    V <- diag(exp(X[, t+1]/2))
    Y[, t] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
  }
  
  multi_sv_data <- list(X = X, Y = Y, Phi = Phi, Sigma_eta = Sigma_eta, 
                        Sigma_eps = Sigma_eps)
  
  if (save_data) {
    saveRDS(multi_sv_data, file = paste0("./data/multi_sv_data_Tfin", Tfin, "_", date, "_", dataset, ".rds"))
  }
} else {
  multi_sv_data <- readRDS(file = paste0("./data/multi_sv_data_Tfin", Tfin, "_", date, "_", dataset, ".rds"))
  
  X <- multi_sv_data$X
  Y <- multi_sv_data$Y
  Phi <- multi_sv_data$Phi
  Sigma_eta <- multi_sv_data$Sigma_eta
  Sigma_eps <- multi_sv_data$Sigma_eps
}

par(mfrow = c(2,1))
plot(X[1, ], type = "l")
plot(X[2, ], type = "l")

plot(Y[1, ], type = "l")
plot(Y[2, ], type = "l")


### Test autocovariance function for VAR(1) ##
Sigma1 <- autocov_VAR1(Phi, Sigma_eta, 0)
# Sigma1_test <- compute_autocov_VAR1(Phi, Sigma_eta)

## Construct initial distribution/prior
prior <- construct_prior(data = Y, byrow = F)
prior_mean <- prior$prior_mean
diag_prior_var <- diag(prior$prior_var)

### STAN ###

hmc_filepath <- paste0(result_directory, "hmc_results_Tfin", Tfin, 
                       "_", date, "_", dataset, ".rds")

if (rerun_hmc) {
  result_directory <- "./results/"
  
  hmc_filepath <- paste0(result_directory, "hmc_results_Tfin", Tfin, 
                         "_", date, "_0", ".rds")
  
  n_post_samples <- 10000
  burn_in <- 1000
  stan.iters <- n_post_samples + burn_in
  
  stan_file <- "./source/stan_multi_sv.stan"
  
  multi_sv_model <- cmdstan_model(
    stan_file,
    cpp_options = list(stan_threads = TRUE)
  )
  
  # multi_sv_data <- list(d = nrow(Y), Tfin = ncol(Y), Y = Y,
  #                       prior_mean_A = prior_mean[1:4], prior_var_A = prior_var[1:4, 1:4],
  #                       prior_mean_gamma = prior_mean[5:6], prior_var_gamma = prior_var[5:6, 5:6]
  # )
  
  multi_sv_data <- list(d = nrow(Y), Tfin = ncol(Y), Y = Y,
                        prior_mean_A = prior_mean[1:4], diag_prior_var_A = diag_prior_var[1:4],
                        prior_mean_gamma = prior_mean[5:6], diag_prior_var_gamma = sqrt(diag_prior_var[5:6])
  )
  
  
  fit_stan_multi_sv <- multi_sv_model$sample(
    multi_sv_data,
    chains = 1,
    threads = parallel::detectCores(),
    refresh = 5,
    iter_warmup = burn_in,
    iter_sampling = n_post_samples
  )
  
  stan_results <- fit_stan_multi_sv$draws(variables = c("Phi_mat", "Sigma_eta_mat"))
  
  if (save_hmc_results) {
    saveRDS(stan_results, hmc_filepath)
  }
  
} else {
  stan_results <- readRDS(hmc_filepath)
}

hmc.post_samples_Phi <- stan_results[,,1:4]
hmc.post_samples_Sigma_eta <- stan_results[,,5:8]

par(mfrow = c(2,4))
plot(density(hmc.post_samples_Phi[,,1]), main = "phi_11")
abline(v = Phi[1,1], lty = 2)

plot(density(hmc.post_samples_Phi[,,2]), main = "phi_21")
abline(v = Phi[2,1], lty = 2)

plot(density(hmc.post_samples_Phi[,,3]), main = "phi_12")
abline(v = Phi[1,2], lty = 2)

plot(density(hmc.post_samples_Phi[,,4]), main = "phi_22")
abline(v = Phi[2,2], lty = 2)

plot(density(hmc.post_samples_Sigma_eta[,,1]), main = "Sigma_eta_11")
abline(v = Sigma_eta[1,1], lty = 2)

# plot(density(hmc.post_samples_Sigma_eta[,,2]), main = "Sigma_eta_21")
# abline(v = Sigma_eta[2,1], lty = 2)
# 
# plot(density(hmc.post_samples_Sigma_eta[,,3]), main = "Sigma_eta_12")
# abline(v = Sigma_eta[1,2], lty = 2)

plot(density(hmc.post_samples_Sigma_eta[,,4]), main = "Sigma_eta_22")
abline(v = Sigma_eta[2,2], lty = 2)

## Trace plots
par(mfrow = c(3,2))
plot(hmc.post_samples_Phi[,,1], type = "l", main = "phi_11")
plot(hmc.post_samples_Phi[,,2], type = "l", main = "phi_21")
plot(hmc.post_samples_Phi[,,3], type = "l", main = "phi_12")
plot(hmc.post_samples_Phi[,,4], type = "l", main = "phi_22")

plot(hmc.post_samples_Sigma_eta[,,1], type = "l", main = "Sigma_eta_11")
# plot(hmc.post_samples_Sigma_eta[,,2], type = "l", main = "Sigma_eta_21")
# plot(hmc.post_samples_Sigma_eta[,,3], type = "l", main = "Sigma_eta_12")
plot(hmc.post_samples_Sigma_eta[,,4], type = "l", main = "Sigma_eta_22")

# multi_sv_code <- "
#     data {
#       int d;    // dimension of the data at time t
#       int<lower=0> Tfin;   // # time points (equally spaced)
#       matrix[d, Tfin] Y;
#       matrix[d, d] Sigma_1;
#     }
#     parameters {
#       matrix[d, d] A;
#       //real gamma_11;
#       //real gamma_22;
#       vector[d] gamma;
#       matrix[d, Tfin] X;
#     }
#     transformed parameters { // define the mapping from A to Phi here
#       matrix[d, d] Phi_mat;
#       matrix[d, d] Sigma_eta_mat;
#       matrix[d, d] B;
#       matrix[d, d] P1;
#       matrix[d, d] Sigma_tilde;
#       matrix[d, d] P;
#       matrix[d, d] Q;
#       matrix[d, d] T;
#       //matrix[d*d, d*d] Phi_kron;
#       //vector[d*d] vec_Sigma1;
#       //matrix[d, d] Sigma1;
# 
#       //
#       Sigma_eta_mat = diag_matrix(exp(gamma));
# 
#       // Autocovariance matrix for VAR(1)
#       //Phi_kron = kronecker_prod(Phi_mat, Phi_mat)
#       //vec_Sigma1 = inverse(diag_matrix(rep_vector(1.0, d^2)) - Phi_kron) * to_vector_colwise(Sigma_eta_mat)
#       //Sigma1 = to_matrix_colwise(vec_Sigma1, 2, 2)
# 
#       // Transition matrix
#       B = cholesky_decompose(diag_matrix(rep_vector(1.0, d)) + A * A');
#       P1 = inverse(B) * A; // B \ A; // same as inv(B) * A
#       Sigma_tilde = diag_matrix(rep_vector(1.0, d)) - P1 * P1';
#       P = cholesky_decompose(Sigma_tilde);
#       Q = cholesky_decompose(Sigma_eta_mat);
#       T = Q / P; // same as Q * inv(P)
#       Phi_mat = T * P1 * inverse(T);
#     }
#     model {
#       to_vector(A) ~ normal(0, 1);
#       gamma ~ normal(0, 0.1);
# 
#       X[, 1] ~ multi_normal(rep_vector(0, d), arima_stationary_cov(Phi_mat,
#                                                 cholesky_decompose(Sigma_eta_mat)))
#       //X[, 1] ~ multi_normal(rep_vector(0, d), Sigma_1);
#       //X[, 1] ~ normal(rep_vector(0, d), 1);
#       for (t in 2:Tfin)
#         X[, t] ~ multi_normal(Phi_mat * X[, t-1], Sigma_eta_mat);
#       for (t in 1:Tfin)
#         Y[, t] ~ multi_normal(rep_vector(0, d),  diag_matrix(exp(X[, t]/2)));
#     }
#   "
# 
# multi_sv_data <- list(d = nrow(Y), Tfin = ncol(Y), Y = Y, Sigma_1 = diag(2))
# 
# if (rerun_hmc) {
#   # hfit <- run_hmc_sv(data = Y, iters = stan.iters, burn_in = burn_in)
# 
#   hfit <- stan(model_code = multi_sv_code,
#                model_name = "multi_sv", data = multi_sv_data,
#                iter = stan.iters, warmup = burn_in, chains=1)
# 
#   hmc.fit <- extract(hfit, pars = c("A", "gamma"), permuted = F)
# 
#   if (save_hmc_results) {
#     saveRDS(hmc.fit, hmc_filepath)
#   }
# } else {
#   hmc.fit <- readRDS(hmc_filepath)
# }

# A11 <- hmc.fit[,,1]
# A12 <- hmc.fit[,,2]
# A21 <- hmc.fit[,,3]
# A22 <- hmc.fit[,,4]
# 
# gamma11 <- hmc.fit[,,5]
# gamma22 <- hmc.fit[,,6]
# 
# hmc.post_samples_A <- list()
# hmc.post_samples_Sigma_eta <- list()
# 
# for (i in 1:length(A11)) {
#   hmc.post_samples_A[[i]] <- matrix(c(A11[i], A12[i], A21[i], A22[i]), 2, 2, byrow = T)
#   hmc.post_samples_Sigma_eta[[i]] <- diag(exp(c(gamma11[i], gamma22[i])))
# }
# 
# hmc.post_samples_Phi <- mapply(backward_map, hmc.post_samples_A, hmc.post_samples_Sigma_eta,
#                                SIMPLIFY = F)
# 
# 
# hmc.post_samples_phi_11 <- lapply(hmc.post_samples_Phi, function(x) x[1,1])
# hmc.post_samples_phi_12 <- lapply(hmc.post_samples_Phi, function(x) x[1,2])
# hmc.post_samples_phi_21 <- lapply(hmc.post_samples_Phi, function(x) x[2,1])
# hmc.post_samples_phi_22 <- lapply(hmc.post_samples_Phi, function(x) x[2,2])
# 
# hmc.post_samples_phi_11 <- as.mcmc(unlist(hmc.post_samples_phi_11))
# hmc.post_samples_phi_12 <- as.mcmc(unlist(hmc.post_samples_phi_12))
# hmc.post_samples_phi_21 <- as.mcmc(unlist(hmc.post_samples_phi_21))
# hmc.post_samples_phi_22 <- as.mcmc(unlist(hmc.post_samples_phi_22))
# 
# hmc.post_samples_sigma_11 <- lapply(hmc.post_samples_Sigma_eta, function(x) x[1,1])
# hmc.post_samples_sigma_22 <- lapply(hmc.post_samples_Sigma_eta, function(x) x[2,2])
# 
# hmc.post_samples_sigma_11 <- as.mcmc(unlist(hmc.post_samples_sigma_11))
# hmc.post_samples_sigma_22 <- as.mcmc(unlist(hmc.post_samples_sigma_22))
# 
# par(mfrow = c(3,2))
# plot(density(hmc.post_samples_phi_11), main = "Posterior of phi_11")
# abline(v = Phi[1,1], lty = 2)
# 
# plot(density(hmc.post_samples_phi_12), main = "Posterior of phi_12")
# abline(v = Phi[1,2], lty = 2)
# 
# plot(density(hmc.post_samples_phi_21), main = "Posterior of phi_12")
# abline(v = Phi[2,1], lty = 2)
# 
# plot(density(hmc.post_samples_phi_22), main = "Posterior of phi_12")
# abline(v = Phi[2,2], lty = 2)
# 
# plot(density(hmc.post_samples_sigma_11), main = "Posterior of sigma_eta_11")
# abline(v = Sigma_eta[1,1], lty = 2)
# 
# plot(density(hmc.post_samples_sigma_22), main = "Posterior of sigma_eta_22")
# abline(v = Sigma_eta[2,2], lty = 2)
# 
# # traceplot(hfit, c("A", "gamma"),
# #           ncol=2,nrow=3,inc_warmup=F)
# 
# 
# # A <- forward_map(Phi, Sigma_eta)
# # Phi_old <- backward_map(A, Sigma_eta)
# # Phi_new <- backward_map2(A, Sigma_eta)
# 

## Test Kronecker funciton here

# kronecker_prod <- function (A, B) {
#   # matrix[rows(A) * rows(B), cols(A) * cols(B)] C;
#   # int m;
#   # int n;
#   # int p;
#   # int q;
#   m = rows(A)
#   n = cols(A)
#   p = rows(B)
#   q = cols(B)
#   for (i in 1:m) {
#     for (j in 1:n) {
#       # int row_start;
#       # int row_end;
#       # int col_start;
#       # int col_end;
#       row_start = (i - 1) * p + 1;
#       row_end = (i - 1) * p + p;
#       col_start = (j - 1) * q + 1;
#       col_end = (j - 1) * q + q;
#       C[row_start:row_end, col_start:col_end] = A[i, j] * B;
#     }
#   }
#   return(C)
# }
# 
# kronecker(matrix(1:4, 2, 2), matrix(1:4, 2, 2))
# 
# ## Test arima_autocov function here
# arima_stationary_cov <- function(Phi, Sigma_eta) {
#   # matrix[rows(Phi), cols(Phi)] Sigma1;
#   # matrix[rows(Phi) * rows(Phi), rows(Phi) * rows(Phi)] Phi_kron;
#   # vector[rows(Phi) * rows(Phi)] vec_Sigma1;
#   # int m;
#   # int m2;
#   m = dim(Phi)[1]
#   m2 = m * m
#   Phi_kron = kronecker(Phi, Phi)
#   Sigma1 = matrix(solve(diag(m2) - Phi_kron) %*% c(Sigma_eta), m, m);
#   return(Sigma1)
# }
# 
# Sigma1_test <- arima_stationary_cov(Phi, Sigma_eta)

# ## Test transition matrix
# to_VAR1_trans_mat <- function(A, Sigma_eta) {
#   # //int d;
#   # //d = rows(A);
#   # matrix[rows(A), cols(A)] Phi_mat;
#   # //matrix[d, d] Sigma_eta_mat;
#   # matrix[rows(A), cols(A)] B;
#   # matrix[rows(A), cols(A)] P1;
#   # matrix[rows(A), cols(A)] Sigma_tilde;
#   # matrix[rows(A), cols(A)] P;
#   # matrix[rows(A), cols(A)] Q;
#   # matrix[rows(A), cols(A)] T;
#   # int d;
#   d = dim(A)[1]
#   B = t(chol(diag(d) + A %*% t(A)))
#   P1 = solve(B) %*% A #// B \ A; // same as inv(B) * A
#   Sigma_tilde = diag(d) - P1 %*% t(P1)
#    P = t(chol(Sigma_tilde))
#    Q = t(chol(Sigma_eta))
#    Tmat = Q %*% solve(P)
#    Phi_mat = Tmat %*% P1 %*% solve(Tmat)
#    return (Phi_mat)
# }
# 
# A <- forward_map(Phi, Sigma_eta)
# Phi_hmc <- to_VAR1_trans_mat(A, Sigma_eta)
# Phi_R <- backward_map(A, Sigma_eta)
