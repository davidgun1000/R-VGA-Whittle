data {
  int<lower=0> Tfin;   // # time points (equally spaced)
  vector[Tfin] y;      // log-squared, mean-removed series
}
parameters {
  real theta_phi;
  real theta_sigma_eta;
  real theta_sigma_eps;
  vector[Tfin] x;                 // log volatility at time t
}
transformed parameters {
  real<lower = -1, upper = 1> phi;
  real<lower = 0> sigma_eta;
  real<lower = 0> sigma_eps;
  
  phi = tanh(theta_phi);
  sigma_eta = sqrt(exp(theta_sigma_eta));
  sigma_eps = sqrt(exp(theta_sigma_eps));
  
}
model {
  theta_phi ~ normal(0, 1);
  theta_sigma_eta ~ normal(0, 1);
  theta_sigma_eps ~ normal(0, 1);
  
  x[1] ~ normal(0, sqrt(sigma_eta^2 / (1 - phi^2)));
  for (t in 2:Tfin)
    x[t] ~ normal(phi * x[t - 1], sigma_eta);
  for (t in 1:Tfin)
    y[t] ~ normal(x[t], sigma_eps);
  
}