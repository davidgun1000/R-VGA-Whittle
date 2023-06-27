## MCMC with Whittle likelihood
library(stats)
library(LSTS)
library(coda)
library(Matrix)
source("calculate_likelihood.R")

## Flags
date <- "20230417" # 20240410 has phi = 0.9
regenerate_data <- F
rerun_mcmc <- T
adapt_proposal <- T
use_whittle_likelihood <- F
save_data <- F
save_results <- F
save_plots <- F

## Parameters 
phi <- 0.9
sigma_e <- 0.5
n <- 10000 # time series length

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

plot(1:n, x, type = "l")

# calculate_whittle_likelihood <- function(series, phi, sigma_e) {
#   n <- length(series)
#   
#   ## Calculate the spectral density of an AR(1) -- turn this into a function later
#   k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
#   k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
#   freq <- 2 * pi * k_in_likelihood / n
#   # spectral_dens <- 1/(2*pi) * sigma_e^2 / (1 + phi^2 - 2 * phi * cos(freq))
#   spectral_dens <- sigma_e^2 / (1 + phi^2 - 2 * phi * cos(freq))
#   
#   ## Spectral density (code from LSTS)
#   # n <- length(series)
#   # m <- n/2
#   # N <- sum(is.na(series))
#   # series[is.na(series)] <- 0
#   # aux <- Mod(fft(series))^2
#   # periodogram <- (aux[2:(m + 1)])/(2 * pi * (n - N))
#   # include.taper <- F
#   # if (include.taper == TRUE) {
#   #   periodogram <- (aux[2:(m + 1)])/(3 * pi * (n - N)/4)
#   # }
#   # lambda <- (2 * pi * (1:m))/n
#   # f <- spectral.density(ar = phi, d = 0,
#   #                       sd = sigma_e, lambda = lambda)
#   # spectral_dens2 <- f[1:(length(f)-1)]
#   
#   ## Fourier transform of the series
#   fourier_transf <- fft(x)
#   I_omega <- 1/n * Mod(fourier_transf)^2
#   
#   plot(I_omega[1:(n/2)], type = "l")
#   browser()
#   
#   # J <- c()
#   # for (j in 1:length(freq)) {
#   #   omega <- freq[j]
#   #   t <- 1:n
#   #   terms <- series * exp(-1i * omega * t)
#   #   J[j] <- sum(terms) #1/sqrt(2*pi) * sum(terms)
#   # }
#   # I_omega2 <- 1/n * Mod(J)^2
#   
#   ## Calculate the Whittle likelihood
#   part1 <- log(spectral_dens)
#   # part2 <- I_omega[k_in_likelihood + 1] /(n/2 * spectral_dens)
#   part2 <- I_omega[k_in_likelihood + 1] /(spectral_dens)
#   
#   # part2 <- I_omega/spectral_dens
#   
#   log_whittle <- - sum(part1 + part2)
#   
#   return(log_whittle)
# }
# 
# ## Test by comparing with the usual likelihood (use KF or EnKF here)
# calculate_ar1_likelihood <- function(series, phi, sigma_e) {
#   log_likelihood <- c()
#   log_likelihood[1] <- dnorm(x[1], 0, sqrt(sigma_e^2 / (1 - phi^2)), log = T) # assume x1 ~ U[-10, 10]
#   for (t in 2:n) {
#     log_likelihood <- log_likelihood + dnorm(series[t], phi * series[t-1], sigma_e, log = T)
#   }
#   return(log_likelihood)
# }

whittle.loglik<-function(series, phi, sigma_e)  	
{  	
  #	
  #   Whittle Loglikelihood	
  #	
  series <- series - mean(series)  	
  a <- fft(series)  	
  a <- Mod(a)^2  	
  n <- length(series)  	
  a <- a/(2 * pi * n)  	
  m <- n/2  	
  #  	
  #Fourier frequencies  	
  #  	
  w <- (2 * pi * (1:m))/n  	
  #  	
  #Spectral Density:  	
  #  	
  # b <- fn.density(w, x)  	
  b <- sigma_e^2 / (1 + phi^2 - 2 * phi * cos(2*pi*w))
  #  	
  #  Calculate sigma^2  	
  #  	
  sigma2 <- (2 * sum(a[1:m]/b))/n  	
  #  	
  #  Whittle Log-likelihood  	
  #  	
  loglik <- 2 * pi * (sum(log(b)) + sum(a[1:m]/b)/sigma2)  	
  return(loglik/n + pi * log(sigma2))  	
}  





# Plot whittle likelihood for various parameter values
phi_grid <- seq(-0.99, 0.99, length.out = 200)
whittle_llh <- c()
whittle_llh2 <- c()
whittle_llh3 <- c()

normal_llh <- c()
for (p in 1:length(phi_grid)) {
  whittle_llh[p] <- calculate_whittle_likelihood(series = x, phi = phi_grid[p], sigma_e = sigma_e)
  # whittle_llh2[p] <- whittle.loglik(series = x, phi = phi_grid[p], sigma_e = sigma_e)
  whittle_llh3[p] <- LS.whittle.loglik(x = c(phi_grid[p], 0, sigma_e), series = x, order = c(p = 1, q = 0),
                                    ar.order = 1)

  normal_llh[p] <- calculate_ar1_likelihood(series = x, phi = phi_grid[p], sigma_e = sigma_e)
}
# 
par(mfrow = c(1, 2))
plot(phi_grid, whittle_llh, type = "l", main = "Whittle likelihood")
lines(phi_grid, whittle_llh3, col = "blue")
abline(v = phi, col = "red", lty = 2)
plot(phi_grid, normal_llh, type = "l", main = "Normal likelihood")
abline(v = phi, col = "red", lty = 2)

################################################################################
##        Now pretend we didn't know phi and try to recover it with MCMC
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

MCMC_iters <- 20000
acceptProb <- c()
accept <- rep(0, MCMC_iters)
post_samples <- c()

## Adapting proposal
D <- 1
# stepsize <- 0.02
scale <- 1
target_accept <- 0.23

## Prior: theta = tanh^-1(phi), theta ~ N(0,1)
prior_mean <- 0
prior_var <- 1
theta <- rnorm(10000, prior_mean, sqrt(prior_var))
phi_samples <- tanh(theta)
hist(phi_samples, main = "Samples from the prior of phi")

## ALternatively can use theta = logit(phi)
# phi_samples2 <- exp(theta) / (1 + exp(theta))
# hist(phi_samples2)

## Initial theta: 
theta_0 <- 1
theta_curr <- theta_0
phi_curr <- tanh(theta_curr)

if (rerun_mcmc) {
  
  mcmc.t1 <- proc.time()
  
  ## Compute initial likelihood
  if (use_whittle_likelihood) {
    # log_likelihood_curr <- - LS.whittle.loglik(x = c(phi_curr, 0, sigma_e), 
    #                                            series = x, order = c(p = 1, q = 0),
    #                                            ar.order = 1)
    log_likelihood_curr <- calculate_whittle_likelihood(series = x, 
                                                        phi = phi_curr, 
                                                        sigma_e = sigma_e)
    
  } else {
    t1 <- proc.time()
    log_likelihood_curr <- calculate_ar1_likelihood(series = x, 
                                                    phi = phi_curr,
                                                    sigma_e = sigma_e)
    t2 <- proc.time()
    
    # log_likelihood_curr2 <- calculate_ar1_likelihood(series = x,
    #                                                 phi = phi_curr,
    #                                                 sigma_e = sigma_e)
    # t3 <- proc.time()
  }
  
  ## Compute initial prior
  log_prior_curr <- dnorm(theta_0, prior_mean, sqrt(prior_var), log = T)
  
  for (i in 1:MCMC_iters) {
    
    ## 1. Propose new parameter values
    theta_prop <- rnorm(1, theta_curr, sqrt(D))
    phi_prop <- tanh(theta_prop)
    
    if (i %% (MCMC_iters/100) == 0) {
      cat(i/MCMC_iters * 100, "% complete \n")
      cat("Proposed phi =", tanh(theta_prop), ", Current param:", tanh(theta_curr), "\n")
      cat("------------------------------------------------------------------\n")
    }
    
    ## 2. Calculate likelihood
    if (use_whittle_likelihood) {
      # log_likelihood_prop <- - LS.whittle.loglik(x = c(phi_prop, 0, sigma_e), 
      #                                            series = x, order = c(p = 1, q = 0),
      #                                            ar.order = 1)
      
      log_likelihood_prop <- calculate_whittle_likelihood(series = x, 
                                                          phi = phi_prop, 
                                                          sigma_e = sigma_e)
      
    } else { # use Gaussian likelihood
      log_likelihood_prop <- calculate_ar1_likelihood(series = x, 
                                                      phi = phi_prop,
                                                      sigma_e = sigma_e)
    }
    
    ## 3. Calculate prior
    log_prior_prop <- dnorm(theta_prop, prior_mean, sqrt(prior_var), log = T)
    
    ## 4. Calculate acceptance probability
    if (abs(tanh(theta_prop)) < 1) { ## if the proposed phi is within (-1, 1)
      r <- exp((log_likelihood_prop + log_prior_prop) - (log_likelihood_curr + log_prior_curr))
      a <- min(1, r)
    } else {
      a <- 0
    }
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
      theta_curr <- theta_prop
      log_likelihood_curr <- log_likelihood_prop
      log_prior_curr <- log_prior_prop
    } 
    
    ## Store parameter 
    post_samples[i] <- phi_curr #theta_curr
    
    ## Adapt proposal covariance matrix
    if (adapt_proposal) {
      if ((i >= 200) && (i %% 10 == 0)) {
        scale <- update_sigma(scale, a, target_accept, i, 1)
        D <- scale * var((post_samples[1:i]))
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

# Burn-in about 1000 seems OK.
burn_in <- 0.2 * MCMC_iters
draws = as.mcmc(draws[-(1:burn_in)])

# Trace plot
traceplot(draws, main = "Trace plot")

## Compare with exact likelihood
# gaussian_mcmc_results <- readRDS(file = paste0("./results/mcmc_exact_results_n", n, "_phi", phi_string, "_", date, ".rds"))
# gaussian_mcmc_post_samples <- gaussian_mcmc_results$post_samples[-(1:burn_in)]

plot(density(draws), xlab = "phi", #xlim = c(0.88, 0.92), 
     col = "blue", main = paste0("Posterior draws (series length = ", n, ")"), lwd = 2)
abline(v = phi, col = "black", lty = 2, lwd = 2)
legend("topleft", legend = c("MCMC Whittle", "MCMC Gauss"), 
       col = c("red", "blue"), lty = 1, lwd = 2)

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
autocorr(draws, lags = c(0, 1, 5, 10, 50), relative=TRUE)
lag.max <- 20

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

