run_hmc_multi_sv <- function(data, transform = "logit", iters, burn_in) {
  
  y <- data
  
  sv_code <- '
    data {
      int<lower=0> Tfin;   // # time points (equally spaced)
      vector[Tfin] y;      // log-squared, mean-removed series
      real kappa;
    }
    parameters {
      real theta_phi;
      real theta_sigma;
      vector[Tfin] x;                 // log volatility at time t
    }
    model {
      theta_phi ~ normal(0, 1);
      theta_sigma ~ normal(0, 1);
      
      x[1] ~ normal(0, sqrt(exp(theta_sigma) / (1 - (tanh(theta_phi)^2))));
      for (t in 2:Tfin)
        x[t] ~ normal(tanh(theta_phi) * x[t - 1], sqrt(exp(theta_sigma)));
      for (t in 1:Tfin)
        y[t] ~ normal(0, kappa * exp(x[t]/2));
    }
  '
  
  log_kappa2_est <- mean(log(y^2)) - (digamma(1/2) + log(2))
  
  sv_data <- list(Tfin = length(y), y = y,
                  kappa = sqrt(exp(log_kappa2_est)))
  
  hfit <- stan(model_code = sv_code, 
               model_name="sv", data = sv_data, 
               iter = iters, warmup = burn_in, chains=1)
  
  return(hfit)
}