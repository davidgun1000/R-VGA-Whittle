setwd("/home/babv971/R-VGA-Whittle/LGSS/")

library(mvtnorm)
library(tensorflow)
reticulate::use_condaenv("myenv", required = TRUE)
library(keras)

source("hmc_functions.R")

date <- "20230525"
regenerate_data <- T
save_data <- F

result_directory <- "./results/"


## True parameters
sigma_eps <- 0.5 # measurement error var
sigma_eta <- 0.7 # process error var
phi <- 0.7

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

# Generate true process x_1:T
n <- 10000
# times <- seq(0, 1, length.out = iters)

lgss_data <- NULL
if (regenerate_data) {
  print("Generating data...")
  x <- c()
  x[1] <- 1
  set.seed(2023)
  for (t in 2:n) {
    x[t] <- phi * x[t-1] + rnorm(1, 0, sigma_eta)
  }
  
  # Generate observations y_1:T
  y <- x + rnorm(n, 0, sigma_eps)
  
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

d <- 3

## Prior
prior_mean <- rep(0, d)
prior_var <- diag(d)

## Transform data to frequency domain
## Fourier frequencies
k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
freq <- 2 * pi * k_in_likelihood / n

## Fourier transform of the observations
fourier_transf <- fft(y)
periodogram <- 1/n * Mod(fourier_transf)^2
I <- periodogram[k_in_likelihood + 1]

# To compute the likelihood
# params_list <- list(phi = phi_grid[i], sigma_eta = sigma_eta, sigma_eps = sigma_eps)
# if (use_whittle_likelihood) {

# TEST IF THE LIKELIHOOD COMPUTATIONS USING TF AND THE USUAL NONTF FUNCTIONS AGREE
# params_list <- c(phi, sigma_eta, sigma_eps)
# likelihood1 <- compute_whittle_likelihood_lgss(y = y, params = params_list)
# likelihood2 <- compute_grad_arctanh(tf$Variable(params_list), I, freq)

# phi_grid <- seq(0.1, 0.99, length.out = 100) DISCARD THIS PART
# llh1 <- c()
# llh2 <- c()
# for (i in 1:length(phi_grid)) {
#   params_list <- c(phi_grid[i], sigma_eta, sigma_eps)
#   llh1[i] <- compute_whittle_likelihood_lgss(y = y, params = params_list)
#   llh2[i] <- as.vector(compute_grad_arctanh(tf$Variable(params_list), I, freq)$log_likelihood)
  
# }
# plot(phi_grid, llh2, type = "l")
# lines(phi_grid, llh1, col = "blue")
# abline(v = phi_grid[which.max(llh2)], col = "red")
# abline(v = phi)
# browser()
# ### Model setup -- 2d correlated Gaussian
# S <- matrix(c(1,-0.98,-0.98,1),2,2)
# n <- nrow(S)
# Q <- chol2inv(chol(S))
# cholQ <- chol(Q)

# n <- d # 3 parameters

############# HMC sampler ##############

M <- diag(1,d) # mass matrix

### Sampler parameters -- set eps and L according to eigenvalues of covariance matrix
# E <- eigen(S)
# eps_gen <- function() round(min(E$values),2)
# L = as.integer(max(E$values)/eps_gen())
# print(paste0("eps = ",eps_gen(),". L = ", L))

L <- 10L
eps_gen <- function() runif(1, 0.0104, 0.0156)

# params_ini <- rmvnorm(1, prior_mean, prior_var)
# current_q <- params_ini

# U <- compute_U(current_q, y, prior_mean, prior_var)
# dUdq <- compute_dUdq(current_q, I, freq)

sampler <- hmc_sampler(#U = U, dUdq = dUdq, 
                        M = M, eps_gen = eps_gen, L = L,
                        data = y, I = I, freq = freq,
                        prior_mean = prior_mean, prior_var = prior_var)

# hmc.post_samples <- .hmc_sample (q=current_q, 
#                                   # U=U, dUdq = dUdq, 
#                                   Minv = Minv, cholM = cholM, eps_gen=eps_gen, L=L,lower=lower,upper=upper)

### Now sample
N <- 1000
q <- matrix(0,d,N)
q[,1] <- rnorm(d, prior_mean, prior_var)
for(i in 2:N) q[,i] <- sampler(q = q[,(i-1)])
# plot(t(q),ylim = c(-4,4),xlim=c(-4,4))
#' 
#' 
#' 

