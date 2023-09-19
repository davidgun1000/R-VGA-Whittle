setwd("~/R-VGA-Whittle/Multi_SV/")

library(cmdstanr)

burn_in <- 1000
n_post_samples <- 1000

stan_file <- "./source/stan_mwe.stan"

mwe_model <- cmdstan_model(
  stan_file,
  cpp_options = list(stan_threads = TRUE)
)

# log_kappa2_est <- mean(log(y^2)) - (digamma(1/2) + log(2))

mwe_data <- NULL

# hfit <- stan(model_code = sv_code, 
#              model_name="sv", data = sv_data, 
#              iter = iters, warmup = burn_in, chains=1)

fit_mwe <- mwe_model$sample(
  mwe_data,
  chains = 1,
  threads = parallel::detectCores(),
  refresh = 5,
  iter_warmup = burn_in,
  iter_sampling = n_post_samples
)

mwe_results <- list(draws = fit_stan_multi_sv_whittle$draws(variables = c("phi", "sigma_eta")),
                     time = fit_stan_multi_sv_whittle$time)
