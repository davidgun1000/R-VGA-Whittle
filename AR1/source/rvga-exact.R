
## Check using exact likelihood

## R-VGA with Whittle likelihood (R-VGAW)?
rm(list = ls())

library(stats)
library(LSTS)
library(coda)

source("calculate_likelihood.R")
source("run_mcmc_ar1.R")
result_directory <- "./results/"

## Flags
date <- "20230417" # 20240410 has phi = 0.9, 20230417 has phi = 0.7
regenerate_data <- F
rerun_rvgae <- T
rerun_mcmc_whittle <- F
rerun_mcmc_exact <- F
adapt_proposal <- T
# use_whittle_likelihood <- F
save_data <- F
save_rvgae_results <- F
save_mcmcw_results <- F
save_mcmce_results <- F
save_plots <- F

## R-VGA flags
use_tempering <- T
n_temper <- 1000
temper_schedule <- rep(1/4, 4)

## MCMC flags
MCMC_iters <- 15000
burn_in <- 5000

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
  rvgae_data <- list(x = x, phi = phi, sigma_e = sigma_e)
  
  if (save_data) {
    saveRDS(rvgae_data, file = paste0("./data/ar1_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  }
  
} else {
  rvgae_data <- readRDS(file = paste0("./data/ar1_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  x <- rvgae_data$x
  phi <- rvgae_data$phi
  sigma_e <- rvgae_data$sigma_e
}

plot(1:n, x, type = "l")

###############################################
##        R-VGA with exact likelihood        ##
###############################################
rvgae_filepath <- paste0(result_directory, "rvgae_results_n", n, 
                        "_phi", phi_string, "_", date, ".rds")

S <- 500

mu_0 <- 0
P_0 <- 1

mu_vals <- list()
mu_vals[[1]] <- mu_0

prec <- list()
prec[[1]] <- 1/P_0 #chol2inv(chol(P_0))

# param_dim <- length(mu_0)
# N <- length(y)
# n <- length(y[[1]])

if (rerun_rvgae) {
  
  print("Starting exact R-VGAL...")
  
  rvgae.t1 <- proc.time()
  
  ## Fourier frequencies
  k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
  k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
  freq_in_likelihood <- 2 * pi * k_in_likelihood / n
  
  ## Fourier transform of the series
  fourier_transf <- fft(x)
  periodogram <- 1/n * Mod(fourier_transf)^2
  I <- periodogram[k_in_likelihood + 1]
  
  for (t in 1:length(x)) {
    
    a_vals <- 1
    if (use_tempering) {
      
      if (t <= n_temper) { # only temper the first n_temper observations
        a_vals <- temper_schedule
      } 
    } 
    
    mu_temp <- mu_vals[[t]]
    prec_temp <- prec[[t]] 
    
    for (v in 1:length(a_vals)) {
      
      a <- a_vals[v]
      
      P <- 1/prec_temp
      # P <- chol2inv(chol(prec_temp))
      samples <- rnorm(S, mu_temp, sqrt(P))
      
      grads <- list()
      hessian <- list()
      
      # Calculate Fourier transform of the series here
      
      for (s in 1:S) {
        
        phi_s <- tanh(samples[s])
        
        ## Calculate the spectral density of an AR(1) -- turn this into a function later
        
        if (t == 1) {
          grad <- - phi_s/(1-phi_s^2) + x[t]^2 * phi_s / sigma_e^2
          grad2 <- - (1 + phi_s^2) / (1 - phi_s^2)^2 + x[t]^2 / sigma_e^2
        } else {
          grad <- x[t-1] / sigma_e^2 * (x[t] - phi_s * x[t-1])
          grad2 <- - x[t-1]^2 / sigma_e^2
        }
        
        grads[[s]] <- grad #grad_phi_fd
        hessian[[s]] <- grad2 #grad_phi_2_fd #x
      }
      
      E_grad <- Reduce("+", grads)/ length(grads)
      E_hessian <- Reduce("+", hessian)/ length(hessian)
      
      prec_temp <- prec_temp - a * E_hessian
      # mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * as.matrix(E_grad_logW))  
      mu_temp <- mu_temp + 1/prec_temp * (a * E_grad)
      
    }  
    
    prec[[t+1]] <- prec_temp
    mu_vals[[t+1]] <- mu_temp
    
    if (t %% floor(length(x)/10) == 0) {
      cat(floor(t/length(x) * 100), "% complete \n")
    }
    
  }
  
  rvgae.t2 <- proc.time()
  
  
  ## Plot posterior
  post_var <- solve(prec[[length(freq_in_likelihood)]])
  rvgae.post_samples <- tanh(rnorm(10000, mu_vals[[length(freq_in_likelihood)]], sqrt(post_var))) # these are samples of beta, log(sigma_a^2), log(sigma_e^2)
  plot(density(tanh(rvgae.post_samples)))
  
  ## Save results
  rvgae_results <- list(mu = mu_vals,
                       prec = prec,
                       post_samples = rvgae.post_samples,
                       S = S,
                       use_tempering = use_tempering,
                       temper_schedule = temper_schedule,
                       time_elapsed = rvgae.t2 - rvgae.t1)
  
  if (save_rvgae_results) {
    saveRDS(rvgae_results, rvgae_filepath)
  }
  
} else {
  rvgae_results <- readRDS(rvgae_filepath)
}
rvgae.post_samples <- rvgae_results$post_samples


##############################################
##        MCMC with Whittle likelihood      ##
##############################################

print("Starting MCMC with Whittle likelihood...")
mcmcw_filepath <- paste0(result_directory, "mcmc_whittle_results_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

if (rerun_mcmc_whittle) {
  
  mcmcw_results <- run_mcmc_ar1(series = x, sigma_e =  sigma_e, 
                                iters = MCMC_iters, burn_in = burn_in, 
                                prior_mean = 0, prior_var = 1, 
                                adapt_proposal = T, use_whittle_likelihood = T)
  
  if (save_mcmcw_results) {
    saveRDS(mcmcw_results, mcmcw_filepath)
  }
  
} else {
  mcmcw_results <- readRDS(mcmcw_filepath)
}

mcmcw.post_samples <- as.mcmc(mcmcw_results$post_samples[-(1:burn_in)])

##############################################
##        MCMC with exact likelihood      ##
##############################################
print("Starting MCMC with exact likelihood...")
mcmce_filepath <- paste0(result_directory, "mcmc_exact_results_n", n, 
                         "_phi", phi_string, "_", date, ".rds")

if (rerun_mcmc_exact) {
  
  mcmce_results <- run_mcmc_ar1(series = x, sigma_e =  sigma_e, 
                                iters = MCMC_iters, burn_in = burn_in,
                                prior_mean = 0, prior_var = 1, 
                                adapt_proposal = T, use_whittle_likelihood = F)
  if (save_mcmce_results) {
    saveRDS(mcmce_results, mcmce_filepath)
  }
} else {
  mcmce_results <- readRDS(mcmce_filepath)
}

mcmce.post_samples <- as.mcmc(mcmce_results$post_samples[-(1:burn_in)])

# ## Function to adapt MCMC proposal
# update_sigma <- function(sigma2, acc, p, i, d) { # function to adapt scale parameter in proposal covariance
#   alpha = -qnorm(p/2);
#   c = ((1-1/d)*sqrt(2*pi)*exp(alpha^2/2)/(2*alpha) + 1/(d*p*(1-p)));
#   Theta = log(sqrt(sigma2));
#   Theta = Theta+c*(acc-p)/max(200, i/d);
#   theta = (exp(Theta));
#   theta = theta^2;
#   
#   return(theta)
# }
# 
# if (use_whittle_likelihood) {
#   llh_info <- "whittle"
# } else {
#   llh_info <- "gaussian"
# }
# 
# MCMC_iters <- 10000
# acceptProb <- c()
# accept <- rep(0, MCMC_iters)
# post_samples <- c()
# 
# ## Adapting proposal
# D <- 1
# # stepsize <- 0.02
# scale <- 1
# target_accept <- 0.23
# 
# ## Prior: theta = tanh^-1(phi), theta ~ N(0,1)
# prior_mean <- 0
# prior_var <- 1
# theta <- rnorm(10000, prior_mean, sqrt(prior_var))
# phi_samples <- tanh(theta)
# hist(phi_samples, main = "Samples from the prior of phi")
# 
# ## ALternatively can use theta = logit(phi)
# # phi_samples2 <- exp(theta) / (1 + exp(theta))
# # hist(phi_samples2)
# 
# ## Initial theta: 
# theta_0 <- 1
# theta_curr <- theta_0
# phi_curr <- tanh(theta_curr)
# 
# if (rerun_mcmc) {
#   
#   mcmc.t1 <- proc.time()
#   
#   ## Compute initial likelihood
#   if (use_whittle_likelihood) {
#     log_likelihood_curr <- calculate_whittle_likelihood(series = x, 
#                                                         phi = phi_curr, 
#                                                         sigma_e = sigma_e)
#     
#   } else {
#     log_likelihood_curr <- calculate_ar1_likelihood(series = x, 
#                                                     phi = phi_curr,
#                                                     sigma_e = sigma_e)
#     
#   }
#   
#   ## Compute initial prior
#   log_prior_curr <- dnorm(theta_0, prior_mean, sqrt(prior_var), log = T)
#   
#   for (i in 1:MCMC_iters) {
#     
#     ## 1. Propose new parameter values
#     theta_prop <- rnorm(1, theta_curr, sqrt(D))
#     phi_prop <- tanh(theta_prop)
#     
#     if (i %% (MCMC_iters/10) == 0) {
#       cat(i/MCMC_iters * 100, "% complete \n")
#       cat("Proposed phi =", tanh(theta_prop), ", Current param:", tanh(theta_curr), "\n")
#       cat("------------------------------------------------------------------\n")
#     }
#     
#     ## 2. Calculate likelihood
#     if (use_whittle_likelihood) {
#       log_likelihood_prop <- calculate_whittle_likelihood(series = x, 
#                                                           phi = phi_prop, 
#                                                           sigma_e = sigma_e)
#       
#     } else { # use Gaussian likelihood
#       log_likelihood_prop <- calculate_ar1_likelihood(series = x, 
#                                                       phi = phi_prop,
#                                                       sigma_e = sigma_e)
#     }
#     
#     ## 3. Calculate prior
#     log_prior_prop <- dnorm(theta_prop, prior_mean, sqrt(prior_var), log = T)
#     
#     ## 4. Calculate acceptance probability
#     if (abs(tanh(theta_prop)) < 1) { ## if the proposed phi is within (-1, 1)
#       r <- exp((log_likelihood_prop + log_prior_prop) - (log_likelihood_curr + log_prior_curr))
#       a <- min(1, r)
#     } else {
#       a <- 0
#     }
#     # cat("a =", a, "\n")
#     acceptProb[i] <- a
#     
#     ## 5. Move to next state with acceptance probability a
#     u <- runif(1, 0, 1)
#     if (u < a) {
#       accept[i] <- 1
#       phi_curr <- phi_prop
#       theta_curr <- theta_prop
#       log_likelihood_curr <- log_likelihood_prop
#       log_prior_curr <- log_prior_prop
#     } 
#     
#     ## Store parameter 
#     post_samples[i] <- phi_curr #theta_curr
#     
#     ## Adapt proposal covariance matrix
#     if (adapt_proposal) {
#       if ((i >= 200) && (i %% 10 == 0)) {
#         scale <- update_sigma(scale, a, target_accept, i, 1)
#         D <- scale * var((post_samples[1:i]))
#       }
#     }
#   }
#   
#   mcmc.t2 <- proc.time()
#   
#   ## Save results
#   if (save_mcmc_results) {
#     filepath <- paste0(result_directory, "mcmc_", llh_info, "_results_n", n, "_", date, ".rds")
#     mcmc_results <- list(post_samples = post_samples, phi = phi, sigma_e = sigma_e)
#     saveRDS(mcmc_results, filepath)
#   }
# } else {
#   mcmc_results <- readRDS(file = paste0(result_directory, "mcmc_", llh_info, "_results_n", n, "_", date, ".rds"))
#   post_samples <- mcmc_results$post_samples
# }

# Convert draws into "mcmc" object that CODA likes
# draws <- as.mcmc(tanh(post_samples))
mcmcw.draws <- as.mcmc(mcmcw.post_samples)
mcmce.draws <- as.mcmc(mcmce.post_samples)

###########################################
##       ASSESSING MCMC CONVERGENCE      ##
###########################################
# Cumulative quantiles plot
# ask <- FALSE
# par(mfrow = c(1,1))

# Cumulative posterior quantiles
# cumuplot(mcmcw.draws, probs = c(0.025, 0.5, 0.975), ylab = "", lty = c(2, 1), 
#          lwd = c(1, 2), type = "l", ask, auto.layout = TRUE, col = 1) 
# burn_in <- 0.2 * MCMC_iters
# draws = as.mcmc(draws[-(1:burn_in)])

# Trace plot
traceplot(mcmcw.draws, main = "Trace plot")

## Compare with exact likelihood
# mcmce_results <- readRDS(file = paste0(result_directory, "mcmc_gaussian_results_n", n, "_", date, ".rds"))
# gaussian_mcmc_post_samples <- gaussian_mcmc_results$post_samples[-(1:burn_in)]

margin <- 0.1
plot(density(mcmcw.draws), xlab = "phi", xlim = c(phi - margin, phi + margin), 
     col = "red", main = paste0("Posterior draws (series length = ", n, ")"), lwd = 2)
lines(density(mcmce.draws), col = "blue", lwd = 2)
lines(density(rvgae.post_samples), col = "goldenrod", lwd = 2)
abline(v = phi, col = "black", lty = 2, lwd = 2)
legend("topright", legend = c("MCMC Whittle", "MCMC exact", "R-VGA"), 
       col = c("red", "blue", "goldenrod"), lty = 1, lwd = 2)


if (save_plots) {
  png(paste0("./plots/rvgae_mcmc_whittle_results_n", n, "_phi", phi_string, "_temper500.png"), width = 600, height = 450)
  plot(density(mcmcw.draws), xlab = "phi", xlim = c(phi - margin, phi + margin), 
       col = "red", main = paste0("Posterior draws (series length = ", n, ")"), lwd = 2)
  lines(density(mcmce.draws), col = "blue", lwd = 2)
  lines(density(rvgae.post_samples), col = "goldenrod", lwd = 2)
  abline(v = phi, col = "black", lty = 2, lwd = 2)
  legend("topright", legend = c("MCMC Whittle", "MCMC exact", "R-VGA"), 
         col = c("red", "blue", "goldenrod"), lty = 1, lwd = 2)
  
  dev.off()
}

#######################################
#      MCMC: INSPECT EFFICIENCY      ##
#######################################
# Get autocorrelation from CODA
autocorr(mcmcw.draws, lags = c(0, 1, 5, 10, 50), relative=TRUE)
lag.max <- 20

# ask <- FALSE;
# autocorr.plot(draws, lag.max, auto.layout = TRUE, ask)
# dev.off()

# Effective sample size and inefficiency factors
# Compute effective sample size (ESS). This is of course done after burn-in
ESS <- effectiveSize(mcmcw.draws)
cat("ESS =", ESS)

# Compute Inefficiency factor
# IF <- dim(draws)[1]/ESS
# print(IF)

