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
}

parameters {
  // real theta_phi;
  real theta_sigma;
  
  complex theta_phi;
  //complex z;
  //vector[2] mu;
  //cholesky_cov[2] L_Sigma;
  // ...
}

transformed parameters {
  real<lower = -1, upper = 1> phi;
  real<lower = 0> sigma_eta;
  
  
  phi = tanh(get_real(theta_phi));
  sigma_eta = sqrt(exp(theta_sigma));
}

model {
  vector[nfreq] spec_dens_inv;
  
  // theta_phi ~ normal(0, sqrt(sigma_eta^2/(1 - phi^2)));
  [get_real(theta_phi), get_imag(theta_phi)]' ~ normal(rep_vector(0, 2), rep_vector(sqrt(sigma_eta^2/(1 - phi^2)), 2));
  
  theta_sigma ~ normal(0, 1);

  for (k in 1:nfreq) {
    spec_dens_inv[k] = 1/compute_spec_dens(phi, sigma_eta, freqs[k]);
  }
  
  periodogram ~ exponential(spec_dens_inv); 
}


// data {
//   
// }
// 
// parameters {
//   // real a;
//   // real b;
// }
// 
// model {
//   # complex a = 3.2 + 2i;
//   # complex b = to_complex(3.2, 2);
//   real a;
//   real b;
//   if (a == b) print("hello");
// }