## R-VGA with Whittle likelihood (R-VGAW)?
rm(list = ls())

# library(stats)
# library(LSTS)
# library(coda)

source("calculate_likelihood.R")
source("run_mcmc_ar1.R")
result_directory <- "./results/"

## Flags
date <- "20230417" # 20240410 has phi = 0.9, 20230417 has phi = 0.7
regenerate_data <- F
rerun_rvgae <- T
rerun_vbe <- F
# use_whittle_likelihood <- F
use_matlab_deriv <- T

save_data <- F
save_rvgae_results <- F
save_plots <- F

## R-VGA flags
use_tempering <- T

if (use_tempering) {
  n_temper <- 100
  temper_schedule <- rep(1/10, 10)
  temper_info <- paste0("_temper", n_temper)
} else {
  n_temper <- 0
  temper_schedule <- NULL
  temper_info <- ""
}

## MCMC flags
n_post_samples <- 10000

## Parameters 
phi <- 0.99
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

###############################################
##        R-VGA with exact likelihood        ##
###############################################
rvgae_filepath <- paste0(result_directory, "rvga_exact_results_n", n, 
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
  
  for (t in 1:length(x)) {
    
    a_vals <- 1
    if (use_tempering) {
      
      if (t <= n_temper) { # only temper the first n_temper observations
        a_vals <- temper_schedule
      } 
    } 
    # cat("a_vals =", a_vals, '\n')
    
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
        
        theta_s <- samples[s]
        phi_s <- tanh(theta_s)
        
        ## Calculate the spectral density of an AR(1) -- turn this into a function later
        
        if (t == 1) {
          
          if (use_matlab_deriv) {
            # grad <- - phi_s - (x[t]^2*phi_s*(phi_s^2 - 1))/sigma_e^2
            
            grad <- - tanh(theta_s) - (x[1]^2*tanh(theta_s)*(tanh(theta_s)^2 - 1))/sigma_e^2
            
            grad2 <- tanh(theta_s)^2 + (x[1]^2*(tanh(theta_s)^2 - 1)^2)/sigma_e^2 +
              (2*x[1]^2*tanh(theta_s)^2*(tanh(theta_s)^2 - 1))/sigma_e^2 - 1
            
          } else {
            grad <- - phi_s + x[t]^1 / sigma_e^2 * phi_s * (1 - phi_s^2)
            grad2 <- phi_s^2 - 1 + x[t]^1 / sigma_e^2 * ( (1 - phi_s^2)^2 - 2 * phi_s^2 * (1 - phi_s^2) )
          }
          
        } else {
          
          if (use_matlab_deriv) {
            # grad <- -(x[t-1]*(x[t] - x[t-1]*phi_s)*(phi_s^2 - 1))/sigma_e^2
            
            # grad2 <- (2*x[t-1]*phi_s*(x[t] - x[t-1]*phi_s)*(phi_s^2 - 1))/sigma_e^2 -
            #   (x[t-1]^2*(phi_s^2 - 1)^2)/sigma_e^2
            
            grad <- -(x[t-1]*(x[t] - x[t-1]*tanh(theta_s))*(tanh(theta_s)^2 - 1))/sigma_e^2
            
            
            grad2 <- (2*x[t-1]*tanh(theta_s)*(x[t] - x[t-1]*tanh(theta_s))*(tanh(theta_s)^2 - 1))/sigma_e^2 - 
              (x[t]^2*(tanh(theta_s)^2 - 1)^2)/sigma_e^2
            
            
          } else {
            
            grad <- 1/sigma_e^2 * (x[t] - phi_s * x[t-1]) * x[t-1] * (1 - phi_s^2)
            grad2 <- - x[t-1]^2/sigma_e^2 * (1 - phi_s^2)^2 -
              2 * x[t-1] / sigma_e^2 * phi_s * (1 - phi_s^2) * (x[t] - phi_s * x[t-1])
            
            ## Check 2nd derivative with finite difference
            # incr <- 1e-07
            # theta_add <- theta_s + incr
            # 
            # f_theta <- 1/sigma_e^2 * (x[t] - tanh(theta_s) * x[t-1]) * x[t-1] * (1 - tanh(theta_s)^2)
            # f_theta_add <- 1/sigma_e^2 * (x[t] - tanh(theta_add) * x[t-1]) * x[t-1] * (1 - tanh(theta_add)^2)
            # 
            # grad2_fd <- (f_theta_add - f_theta) / incr
            
          }
          
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
    
    if (prec_temp <= 0) {
      browser()
    }
    
    prec[[t+1]] <- prec_temp
    mu_vals[[t+1]] <- mu_temp
    
    if (t %% floor(length(x)/10) == 0) {
      cat(floor(t/length(x) * 100), "% complete \n")
    }
    
  }
  
  rvgae.t2 <- proc.time()
  
  ## Plot posterior
  post_var <- solve(prec[[length(mu_vals)]])
  rvgae.post_samples <- tanh(rnorm(10000, mu_vals[[length(mu_vals)]], sqrt(post_var))) # these are samples of beta, log(sigma_a^2), log(sigma_e^2)
  # plot(density(rvgae.post_samples))
  
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
##      Batch VB with exact likelihood      ##
##############################################
vbe_filepath <- paste0(result_directory, "vb_exact_results_n", n, 
                       "_phi", phi_string, "_", date, ".rds")

if (rerun_vbe) {
  
} else {
  vbe_results <- readRDS(vbe_filepath)
}
# vbe.post_samples <- rnorm(n_post_samples, 
#                           vbe_results$mu[[length(vbe_results$mu)]],
#                           sqrt(vbe_results$sigma[[length(vbe_results$sigma)]]))

vbe.post_samples <- vbe_results$post_samples


## Plot posterior densities ##
plot(density(vbe.post_samples), col = "goldenrod")
lines(density(rvgae.post_samples), col = "red")
abline(v = phi, lty = 2)
legend("topleft", legend = c("Batch VB exact", "RVGA exact"), col = c("goldenrod", "red"), lty = 1)
