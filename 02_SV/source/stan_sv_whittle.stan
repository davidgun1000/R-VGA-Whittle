functions {
  // Compute spectral density here
  real compute_spec_dens(real phi, real sigma_eta, real freq) {
    real spec_dens;
    spec_dens = sigma_eta^2 / (1 + phi^2 - 2 * phi * cos(freq)) + pi()^2/2;
    return spec_dens;
  }
  
}

data {
  int<lower=0> nfreq;   // # time points (equally spaced)
  vector[nfreq] freqs;
  vector[nfreq] periodogram;
  int transform;
  vector[2] prior_mean;
  vector[2] diag_prior_var;
  // vector[Tfin] y;      // log-squared, mean-removed series
}

parameters {
  real theta_phi;
  real theta_sigma;
}

transformed parameters {
  real<lower = -1, upper = 1> phi;
  real<lower = 0> sigma_eta;
  
  phi = transform ? tanh(theta_phi) : (1 / (1+exp(-theta_phi)));
  //phi = tanh(theta_phi);
  sigma_eta = sqrt(exp(theta_sigma));
  //print("phi = ", phi);
  //print("sigma_eta = ", sigma_eta);
}

model {
  vector[nfreq] spec_dens_inv;
  
  // theta_phi ~ unit_normal();
  // theta_phi ~ normal(0, sqrt(sigma_eta^2/(1 - phi^2)));
  // theta_sigma ~ normal(0, 1);
  
  theta_phi ~ normal(prior_mean[1], sqrt(diag_prior_var[1]));
  theta_sigma ~ normal(prior_mean[2], sqrt(diag_prior_var[2]));

  for (k in 1:nfreq) {
    spec_dens_inv[k] = 1/compute_spec_dens(phi, sigma_eta, freqs[k]);
  }
  
  periodogram ~ exponential(spec_dens_inv); 
}