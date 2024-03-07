data {
  int<lower=1> m; // Dimension of observation vector
  int<lower=1> p; // Order of VAR model
  int<lower=1> N; // Length of time series
  vector[m] y[N]; // Time series
  // Hyperparameters in exchangeable multivariate normal prior for the phi_i
  vector[p] m_diag;
  vector<lower=0>[p] s_diag;
  vector[p] m_offdiag;
  vector<lower=0>[p] s_offdiag;
  // Hyperparameters in exchangeable inverse Wishart prior for Sigma
  real<lower=0> scale_diag;                    // Diagonal element in scale matrix
  real<lower=-scale_diag/(m-1)> scale_offdiag; /* Off-diagonal element in scale 
                                                  matrix */
  real<lower=m+3> df;                          /* Degrees of freedom (limit ensures 
                                                  finite variance) */
}
transformed data {
  vector[m] mu = rep_vector(0.0, m); // (Zero)-mean of VAR process
  matrix[m, m] scale_mat;            // Scale-matrix in prior for Sigma
  for(i in 1:m) {
    for(j in 1:m) {
      if(i==j) scale_mat[i, j] = scale_diag;
      else scale_mat[i, j] = scale_offdiag;
    }
  }
}
parameters {
  matrix[m, m] phi[p]; // The phi_i
  cov_matrix[m] Sigma; // Error variance, Sigma
}
model {
  vector[m] mut_rest[N-p]; // Conditional means of y_{p+1}, ..., y_{N}
  // (Conditional) likelihood (conditional on first p observations):
  for(t in (p+1):N) {
    mut_rest[t-p] = mu;
    for(i in 1:p) {
      mut_rest[t-p] += phi[i] * (y[t-i] - mu);
    }
  }
  y[(p+1):N] ~  multi_normal(mut_rest, Sigma);
  // Prior:
  Sigma ~ inv_wishart(df, scale_mat);
  for(s in 1:p) {
    diagonal(phi[s]) ~ normal(m_diag[s], s_diag[s]);
    for(i in 1:m) {
      for(j in 1:m) {
        if(i != j) phi[s, i, j] ~ normal(m_offdiag[s], s_offdiag[s]);
      }
    }
  }
}
