run_hmc_sv <- function(data, transform = "logit",
                       prior_mean, prior_var, iters, burn_in) {
  
  y <- data
  
  # transform01 <- 0 # 0 == logit, 1 == arctanh
  # if (transform == "arctanh") {
  #   transform01 <- 1
  # }
  # sv_code <- '
  #   data {
  #     int<lower=0> Tfin;   // # time points (equally spaced)
  #     vector[Tfin] y;      // log-squared, mean-removed series
  #     real kappa;
  #   }
  #   parameters {
  #     real theta_phi;
  #     real theta_sigma;
  #     vector[Tfin] x;                 // log volatility at time t
  #   }
  #   model {
  #     theta_phi ~ normal(0, 1);
  #     theta_sigma ~ normal(0, 1);
  #     
  #     x[1] ~ normal(0, sqrt(exp(theta_sigma) / (1 - (tanh(theta_phi)^2))));
  #     for (t in 2:Tfin)
  #       x[t] ~ normal(tanh(theta_phi) * x[t - 1], sqrt(exp(theta_sigma)));
  #     for (t in 1:Tfin)
  #       y[t] ~ normal(0, kappa * exp(x[t]/2));
  #   }
  # '
  
  stan_file <- "./source/stan_sv.stan"
  
  sv_model <- cmdstan_model(
    stan_file,
    cpp_options = list(stan_threads = TRUE)
  )
  
  log_kappa2_est <- mean(log(y^2)) - (digamma(1/2) + log(2))
  
  sv_data <- list(Tfin = length(y), y = y,
                  prior_mean = prior_mean, diag_prior_var = diag(prior_var),
                  kappa = sqrt(exp(log_kappa2_est)), 
                  transform = ifelse(transform == "arctanh", 1, 0))
  
  # hfit <- stan(model_code = sv_code, 
  #              model_name="sv", data = sv_data, 
  #              iter = iters, warmup = burn_in, chains=1)
  
  fit_stan_multi_sv <- sv_model$sample(
    sv_data,
    chains = 1,
    threads = parallel::detectCores(),
    refresh = 100,
    iter_warmup = burn_in,
    iter_sampling = iters
  )
  
  stan_results <- list(draws = fit_stan_multi_sv$draws(variables = c("phi", "sigma_eta")),
                       time = fit_stan_multi_sv$time)
  return(stan_results)
}