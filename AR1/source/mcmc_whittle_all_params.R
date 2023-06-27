## MCMC with Whittle likelihood
library(stats)
library(LSTS)
library(coda)
library(Matrix)
source("./source/calculate_likelihood.R")

## Flags
date <- "20230417" # 20240410 has phi = 0.9
regenerate_data <- F
rerun_mcmc <- T
adapt_proposal <- T
use_whittle_likelihood <- T
save_data <- F
save_results <- F
save_plots <- F

## Parameters 
phi <- 0.9
sigma_e <- 0.5
n <- 1000 # time series length

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

if (regenerate_data) {
  ## Generate AR(1) series
  x0 <- 1
  x <- c()
  x[1] <- x0
  set.seed(2023)
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

# plot(1:n, x, type = "l")

################################################################################
##                      MCMC (inferring phi and sigma_e)                      ##   
################################################################################

## Function to adapt MCMC proposal
update_sigma <- function(sigma2, acc, p, i, d) { # function to adapt scale parameter in proposal covariance
  alpha = -qnorm(p/2);
  c = ((1-1/d)*sqrt(2*pi)*exp(alpha^2/2)/(2*alpha) + 1/(d*p*(1-p)));
  Theta = log(sqrt(sigma2));
  Theta = Theta+c*(acc-p)/max(200, i/d);
  theta = (exp(Theta));
  theta = theta^2;
  
  return(theta)
}

if (use_whittle_likelihood) {
  llh_info <- "whittle"
} else {
  llh_info <- "gaussian"
}

MCMC_iters <- 15000

## Adapting proposal
D <- diag(1, 2)
# stepsize <- 0.02
scale <- 1
target_accept <- 0.23

## Prior: theta = tanh^-1(phi), theta ~ N(0,1)
prior_mean <- c(1, -1)
prior_var <- diag(c(1, 1))
theta <- rmvnorm(10000, prior_mean, prior_var)
phi_samples <- tanh(theta[, 1])
hist(phi_samples, main = "Samples from the prior of phi")
sigma_e_samples <- sqrt(exp(theta[, 1]))
hist(sigma_e_samples, main = "Samples from the prior of sigma_eta")

## ALternatively can use theta = logit(phi)
# phi_samples2 <- exp(theta) / (1 + exp(theta))
# hist(phi_samples2)

acceptProb <- c()
accept <- rep(0, MCMC_iters)
post_samples <- matrix(NA, nrow = MCMC_iters, ncol = length(prior_mean))

## Initial theta: 
theta_0 <- rmvnorm(1, prior_mean, prior_var)
theta_curr <- theta_0
phi_curr <- tanh(theta_curr[1])
sigma_e_curr <- sqrt(exp(theta_curr[2]))

if (rerun_mcmc) {
  
  mcmc.t1 <- proc.time()
  
  ## Compute initial likelihood
  if (use_whittle_likelihood) {
    # log_likelihood_curr <- - LS.whittle.loglik(x = c(phi_curr, 0, sigma_e), 
    #                                            series = x, order = c(p = 1, q = 0),
    #                                            ar.order = 1)
    log_likelihood_curr <- calculate_whittle_likelihood(series = x, 
                                                        phi = phi_curr, 
                                                        sigma_e = sigma_e_curr)
    
  } else {
    t1 <- proc.time()
    log_likelihood_curr <- calculate_ar1_likelihood(series = x, 
                                                    phi = phi_curr,
                                                    sigma_e = sigma_e_curr)
    t2 <- proc.time()
    
    # log_likelihood_curr2 <- calculate_ar1_likelihood(series = x,
    #                                                 phi = phi_curr,
    #                                                 sigma_e = sigma_e)
    # t3 <- proc.time()
  }
  
  ## Compute initial prior
  log_prior_theta_phi_curr <- dnorm(theta_curr[1], prior_mean[1], sqrt(prior_var[1,1]), log = T)
  log_prior_theta_sigma_curr <- dnorm(theta_curr[2], prior_mean[2], sqrt(prior_var[2,2]), log = T)
  log_prior_curr <- log_prior_theta_phi_curr + log_prior_theta_sigma_curr
  
  for (i in 1:MCMC_iters) {
    
    ## 1. Propose new parameter values
    theta_prop <- rmvnorm(1, theta_curr, D)
    phi_prop <- tanh(theta_prop[1])
    sigma_e_prop <- sqrt(exp(theta_prop[2]))
    
    if (i %% (MCMC_iters/10) == 0) {
      cat(i/MCMC_iters * 100, "% complete \n")
      cat("Proposed phi =", tanh(theta_prop[1]), ", Current param:", tanh(theta_curr[1]), "\n")
      cat("------------------------------------------------------------------\n")
    }
    
    ## 2. Calculate likelihood
    if (use_whittle_likelihood) {
      # log_likelihood_prop <- - LS.whittle.loglik(x = c(phi_prop, 0, sigma_e), 
      #                                            series = x, order = c(p = 1, q = 0),
      #                                            ar.order = 1)
      
      log_likelihood_prop <- calculate_whittle_likelihood(series = x, 
                                                          phi = phi_prop, 
                                                          sigma_e = sigma_e_prop)
      
    } else { # use Gaussian likelihood
      log_likelihood_prop <- calculate_ar1_likelihood(series = x, 
                                                      phi = phi_prop,
                                                      sigma_e = sigma_e_prop)
    }
    
    ## 3. Calculate prior
    # log_prior_prop <- dnorm(theta_prop, prior_mean, sqrt(prior_var), log = T)
    log_prior_theta_phi_prop <- dnorm(theta_prop[1], prior_mean[1], sqrt(prior_var[1,1]), log = T)
    log_prior_theta_sigma_prop <- dnorm(theta_prop[2], prior_mean[2], sqrt(prior_var[2,2]), log = T)
    log_prior_prop <- log_prior_theta_phi_prop + log_prior_theta_sigma_prop
    
    ## 4. Calculate acceptance probability
    # if (abs(tanh(theta_prop)) < 1) { ## if the proposed phi is within (-1, 1)
      r <- exp((log_likelihood_prop + log_prior_prop) - (log_likelihood_curr + log_prior_curr))
      a <- min(1, r)
    # } else {
    #   a <- 0
    # }
    # cat("a =", a, "\n")
    acceptProb[i] <- a
    
    ## 5. Move to next state with acceptance probability a
    u <- runif(1, 0, 1)
    if (u < a) {
      accept[i] <- 1
      # phi_curr <- phi_prop
      # sigma2_curr <- sigma2_prop
      # tau2_curr <- tau2_prop
      phi_curr <- phi_prop
      sigma_e_curr <- sigma_e_prop
      theta_curr <- theta_prop
      log_likelihood_curr <- log_likelihood_prop
      log_prior_curr <- log_prior_prop
    } 
    
    ## Store parameter 
    post_samples[i, ] <- c(phi_curr, sigma_e_curr) #theta_curr
    
    ## Adapt proposal covariance matrix
    if (adapt_proposal) {
      if ((i >= 200) && (i %% 10 == 0)) {
        scale <- update_sigma(scale, a, target_accept, i, 1)
        D <- scale * var((post_samples[1:i, ]))
      }
    }
  }
  
  
  mcmc.t2 <- proc.time()
  
  
  ## Save results
  if (save_results) {
    filepath <- paste0("./mcmc_", llh_info, "_results_n", n, "_phi", phi_string, "_", date, ".rds")
    mcmc_results <- list(post_samples = post_samples, phi = phi, sigma_e = sigma_e)
    saveRDS(mcmc_results, filepath)
  }
} else {
  mcmc_results <- readRDS(file = paste0("./mcmc_", llh_info, "_results_n", n, "_phi", phi_string, "_", date, ".rds"))
  post_samples <- mcmc_results$post_samples
}

# Convert draws into "mcmc" object that CODA likes
# draws <- as.mcmc(tanh(post_samples))
draws <- as.mcmc(post_samples)
#######################################################
# ASSESSING MCMC CONVERGENCE
#######################################################
# Cumulative quantiles plot
ask <- FALSE
par(mfrow = c(1,1))

# Cumulative posterior quantiles
cumuplot(draws, probs = c(0.025, 0.5, 0.975), ylab = "", lty = c(2, 1), 
         lwd = c(1, 2), type = "l", ask, auto.layout = TRUE, col = 1) 

# Burn-in 
burn_in <- 5000 #0.2 * MCMC_iters
draws <- as.mcmc(draws[-(1:burn_in), ])
phi_draws <- draws[, 1]
sigma_e_draws <- draws[, 2]

# Trace plot
par(mfrow = c(2, 1))
traceplot(phi_draws, main = "Trace plot of phi")
traceplot(sigma_e_draws, main = "Trace plot of sigma_eta")

## Compare with exact likelihood

plot(density(phi_draws), xlab = "phi", #xlim = c(0.88, 0.92), 
     col = "blue", main = paste0("Posterior draws (series length = ", n, ")"), lwd = 2)
abline(v = phi, col = "black", lty = 2, lwd = 2)
# legend("topleft", legend = c("MCMC Whittle", "MCMC Gauss"), 
#        col = c("red", "blue"), lty = 1, lwd = 2)

plot(density(sigma_e_draws), xlab = "sigma_eta", #xlim = c(0.88, 0.92), 
     col = "blue", main = paste0("Posterior draws (series length = ", n, ")"), lwd = 2)
abline(v = sigma_e, col = "black", lty = 2, lwd = 2)


if (save_plots) {
  png(paste0("rvga_", llh_info, "_results_n", n, ".png"), width = 600, height = 450)
  plot(density(draws), xlab = "phi", #xlim = c(0.88, 0.92), 
       col = "blue", main = paste0("Posterior draws (series length = ", n, ")"), lwd = 2)
  abline(v = phi, col = "black", lty = 2, lwd = 2)
  legend("topleft", legend = c("MCMC Whittle", "MCMC Gauss"), 
         col = c("red", "blue"), lty = 1, lwd = 2)
  
  dev.off()
}
#######################################################
# INSPECT EFFICIENCY 
#######################################################
# Get autocorrelation from CODA
# autocorr(draws, lags = c(0, 1, 5, 10, 50), relative=TRUE)
# lag.max <- 20

# ask <- FALSE;
# pdf(paste(my_dir, 'eBay_Coins_RWMH_autocorr.pdf'))
# autocorr.plot(draws, lag.max, auto.layout = TRUE, ask)
# dev.off()

# Effective sample size and inefficiency factors
# Compute effective sample size (ESS). This is of course done after burn-in
ESS <- effectiveSize(draws)
cat("ESS =", ESS)

# Compute Inefficiency factor
# IF <- dim(draws)[1]/ESS
# print(IF)

