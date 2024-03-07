data {
  int<lower=0> Tfin;   // # time points (equally spaced)
  vector[Tfin] y;      // log-squared, mean-removed series
  real kappa;
  int transform;
  vector[2] prior_mean;
  vector[2] diag_prior_var;
}
parameters {
  real theta_phi;
  real theta_sigma;
  vector[Tfin] x;                 // log volatility at time t
}
transformed parameters {
  real<lower = -1, upper = 1> phi;
  real<lower = 0> sigma_eta;
  
  // if transform != 0 use arctanh, if transform == 0 use logit
  phi = transform ? tanh(theta_phi) : (1 / (1+exp(-theta_phi)));
  sigma_eta = sqrt(exp(theta_sigma));
}
model {
  // theta_phi ~ normal(0, 1);
  // theta_sigma ~ normal(0, 1);
  
  theta_phi ~ normal(prior_mean[1], sqrt(diag_prior_var[1]));
  theta_sigma ~ normal(prior_mean[2], sqrt(diag_prior_var[2]));
  
  // x[1] ~ normal(0, sqrt(exp(theta_sigma) / (1 - (tanh(theta_phi)^2))));
  // for (t in 2:Tfin)
  //   x[t] ~ normal(tanh(theta_phi) * x[t - 1], sqrt(exp(theta_sigma)));
  // for (t in 1:Tfin)
  //   y[t] ~ normal(0, kappa * exp(x[t]/2));
  x[1] ~ normal(0, sqrt(sigma_eta^2 / (1 - phi^2)));
  for (t in 2:Tfin) {
    x[t] ~ normal(phi * x[t - 1], sigma_eta);
  }
  // for (t in 1:Tfin)
  //   y[t] ~ normal(0, kappa * exp(x[t]/2));
  y ~ normal(0, kappa * exp(x / 2));
    
}