## Stan for univariate SV model ##

setwd("~/R-VGA-Whittle/SV")

library("rstan")
library(cmdstanr)

source("./source/run_hmc_sv.R")

## Flags
date <- "20230808"
regenerate_data <- T
save_data <- F

## Generate data
mu <- 0
phi <- 0.9
sigma_eta <- 0.7
sigma_eps <- 1
kappa <- 2
x1 <- rnorm(1, mu, sigma_eta^2 / (1 - phi^2))
n <- 1000

## For the result filename
phi_string <- sub("(\\d+)\\.(\\d+)", "\\1\\2", toString(phi)) ## removes decimal point fron the number

## Generate data
if (regenerate_data) {
  x <- c()
  x[1] <- x1
  
  for (t in 2:(n+1)) {
    x[t] <- phi * x[t-1] + sigma_eta * rnorm(1, 0, 1)
  }
  
  eps <- rnorm(n, 0, sigma_eps)
  y <- kappa * exp(x[2:(n+1)]/2) * eps
  
  par(mfrow = c(1,1))
  plot(x, type = "l")
  
  sv_data <- list(x = x, y = y, phi = phi, sigma_eta = sigma_eta, 
                  sigma_eps = sigma_eps, kappa = kappa)
  
  if (save_data) {
    saveRDS(sv_data, file = paste0("./data/sv_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  }
  
} else {
  print("Reading saved data...")
  sv_data <- readRDS(file = paste0("./data/sv_data_n", n, "_phi", phi_string, "_", date, ".rds"))
  y <- sv_data$y
  x <- sv_data$x
  phi <- sv_data$phi
  sigma_eta <- sv_data$sigma_eta
  sigma_eps <- sv_data$sigma_eps
}

### STAN ###
n_post_samples <- 5000
burn_in <- 1000
stan.iters <- n_post_samples + burn_in

hmc_results <- run_hmc_sv(data = y, iters = stan.iters, burn_in = burn_in)

hmc.phi <- hmc_results$draws[,,1]
hmc.sigma_eta <- hmc_results$draws[,,2]

# par(mfrow = c(1,2))
# plot(density(hmc.phi), main = "Posterior of phi")
# abline(v = phi, lty = 2)
# 
# plot(density(hmc.sigma_eta), main = "Posterior of sigma_eta")
# abline(v = sigma_eta, lty = 2)

# traceplot(hfit, c("theta_phi", "theta_sigma"),
#           ncol=1,nrow=2,inc_warmup=F)

########################################################
##          Stan with the Whittle likelihood          ##
########################################################
## Fourier frequencies
k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
freq <- 2 * pi * k_in_likelihood / n

## Fourier transform of the observations
y_tilde <- log(y^2) - mean(log(y^2))

fourier_transf <- fft(y_tilde)
periodogram <- 1/n * Mod(fourier_transf)^2
I <- periodogram[k_in_likelihood + 1]

whittle_stan_file <- "./source/stan_sv_whittle.stan"
# whittle_stan_file <- "./source/stan_mwe.stan" # this was to test the use of complex numbers

whittle_sv_model <- cmdstan_model(
  whittle_stan_file,
  cpp_options = list(stan_threads = TRUE)
)

# log_kappa2_est <- mean(log(y^2)) - (digamma(1/2) + log(2))

whittle_sv_data <- list(nfreq = length(freq), freqs = freq, periodogram = I)

# hfit <- stan(model_code = sv_code, 
#              model_name="sv", data = sv_data, 
#              iter = iters, warmup = burn_in, chains=1)

fit_stan_multi_sv_whittle <- whittle_sv_model$sample(
  whittle_sv_data,
  chains = 1,
  threads = parallel::detectCores(),
  refresh = 5,
  iter_warmup = burn_in,
  iter_sampling = n_post_samples
)

hmcw_results <- list(draws = fit_stan_multi_sv_whittle$draws(variables = c("phi", "sigma_eta")),
                     time = fit_stan_multi_sv_whittle$time)

# hmcw.fit <- extract(hfit, pars = c("theta_phi", "theta_sigma"),
#                    permuted = F)
# 
hmcw.phi <- hmcw_results$draws[,,1]
hmcw.sigma_eta <- hmcw_results$draws[,,2]

# hmcw.phi <- tanh(hmcw.theta_phi)
# hmcw.sigma_eta <- sqrt(exp(hmcw.theta_sigma))

par(mfrow = c(1,2))
plot(density(hmc.phi), main = "Posterior of phi")
lines(density(hmcw.phi), col = "red")
abline(v = phi, lty = 2)
legend("bottomright", legend = c("HMC", "HMCW"), col = c("black", "red"), lty = 1)

plot(density(hmc.sigma_eta), main = "Posterior of sigma_eta")
lines(density(hmcw.sigma_eta), col = "red")
abline(v = sigma_eta, lty = 2)

# traceplot(hwhittle.fit, c("theta_phi", "theta_sigma"),
#           ncol=1,nrow=2,inc_warmup=F)

