functions {
  // Compute spectral density here
  real compute_spec_dens(real phi, real sigma_eta, real sigma_eps, real freq) {
    real spec_dens;
    spec_dens = sigma_eta^2 / (1 + phi^2 - 2 * phi * cos(freq)) + sigma_eps^2;
    return spec_dens;
  }
}

data {
  int<lower=0> nfreq;   // # time points (equally spaced)
  vector[nfreq] freqs;
  vector[nfreq] periodogram;
  // vector[Tfin] y;      // log-squared, mean-removed series
  // real kappa;
}

parameters {
  real theta_phi;
  real theta_sigma_eta;
  real theta_sigma_eps;
  
  //vector[Tfin] x; 
  // log volatility at time t
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
  vector[nfreq] spec_dens_inv;
  
  theta_phi ~ normal(0, 1);
  theta_sigma_eta ~ normal(0, 1);
  theta_sigma_eps ~ normal(0, 1);
  
  for (k in 1:nfreq) { 
    spec_dens_inv[k] = 1/compute_spec_dens(phi, sigma_eta, sigma_eps, freqs[k]);
  }
  
  periodogram ~ exponential(spec_dens_inv); 
}