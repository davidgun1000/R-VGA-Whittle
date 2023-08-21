## Stan for univariate SV model ##

setwd("~/R-VGA-Whittle/SV")

library("rstan")

source("./source/run_stan_sv.R")
## Flags
date <- "20230808"
regenerate_data <- T
save_data <- F

## Generate data
mu <- 0
phi <- 0.9
sigma_eta <- 0.7
sigma_eps <- 1
kappa <- 10
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
n_post_samples <- 10000
burn_in <- 1000
stan.iters <- n_post_samples + burn_in

# sv_code <- '
#     data {
#       int<lower=0> Tfin;   // # time points (equally spaced)
#       vector[Tfin] y;      // log-squared, mean-removed series
#       real kappa;
#     }
#     parameters {
#       //real mu;                     // mean log volatility
#       //real<lower=-1,upper=1> phi;  // persistence of volatility
#       //real<lower=0> sigma;         // white noise shock scale
#       
#       real theta_phi;
#       real theta_sigma;
#       vector[Tfin] x;                 // log volatility at time t
#     }
#     model {
#       //phi ~ uniform(-1, 1);
#       //sigma ~ cauchy(0, 5);
#       //mu ~ cauchy(0, 10);
#       
#       theta_phi ~ normal(0, 1);
#       theta_sigma ~ normal(0, 1);
#       
#       x[1] ~ normal(0, sqrt(exp(theta_sigma) / (1 - (tanh(theta_phi)^2))));
#       for (t in 2:Tfin)
#         x[t] ~ normal(tanh(theta_phi) * x[t - 1], sqrt(exp(theta_sigma)));
#       for (t in 1:Tfin)
#         y[t] ~ normal(0, kappa * exp(x[t]/2));
#     }
#   '
# 
# log_kappa2_est <- mean(log(y^2)) - (digamma(1/2) + log(2))
# 
# sv_data <- list(Tfin = length(y), y = y,
#                 kappa = sqrt(exp(log_kappa2_est)))
# 
# hfit <- stan(model_code = sv_code, 
#              model_name="sv", data = sv_data, 
#              iter = iters, warmup = burn_in, chains=1)

hfit <- run_hmc_sv(data = y, iters = stan.iters, burn_in = burn_in)

hmc.fit <- extract(hfit, pars = c("theta_phi", "theta_sigma"),
                   permuted = F)

hmc.theta_phi <- hmc.fit[,,1]
hmc.theta_sigma <- hmc.fit[,,2]

hmc.phi <- tanh(hmc.theta_phi)
hmc.sigma_eta <- sqrt(exp(hmc.theta_sigma))

par(mfrow = c(1,2))
plot(density(hmc.phi), main = "Posterior of phi")
abline(v = phi, lty = 2)

plot(density(hmc.sigma_eta), main = "Posterior of sigma_eta")
abline(v = sigma_eta, lty = 2)

traceplot(hfit, c("theta_phi", "theta_sigma"),
          ncol=1,nrow=2,inc_warmup=F)
