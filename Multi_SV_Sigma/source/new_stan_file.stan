functions {
  
  matrix to_matrix_colwise(vector v, int m, int n) {
    matrix[m, n] res;
    for (j in 1:n) {
      for (i in 1:m) {
        res[i, j] = v[(j - 1) * m + m];
      }
    }
    return res;
  }
  
  matrix kronecker_prod(matrix A, matrix B) {
    matrix[rows(A) * rows(B), cols(A) * cols(B)] C;
    int m;
    int n;
    int p;
    int q;
    m = rows(A);
    n = cols(A);
    p = rows(B);
    q = cols(B);
    for (i in 1:m) {
      for (j in 1:n) {
        int row_start;
        int row_end;
        int col_start;
        int col_end;
        row_start = (i - 1) * p + 1;
        row_end = (i - 1) * p + p;
        col_start = (j - 1) * q + 1;
        col_end = (j - 1) * q + q; // the original Stan function had a mistake here; should be +q not +1
        C[row_start:row_end, col_start:col_end] = A[i, j] * B;
      }
    }
    return C;
  }
  
  // Mapping from unconstrained matrix to VAR(1) coefficient matrix
  matrix to_VAR1_trans_mat(matrix A, matrix Sigma_eta) {
    //int d;
    //d = rows(A);
    matrix[rows(A), cols(A)] Phi_mat;
    //matrix[d, d] Sigma_eta_mat;
    matrix[rows(A), cols(A)] B;
    matrix[rows(A), cols(A)] P1;
    matrix[rows(A), cols(A)] Sigma_tilde;
    matrix[rows(A), cols(A)] P;
    matrix[rows(A), cols(A)] Q;
    matrix[rows(A), cols(A)] T;
    int d;
    d = rows(A);
    B = cholesky_decompose(diag_matrix(rep_vector(1.0, d)) + A * A');
    P1 = inverse(B) * A; // B \ A; // same as inv(B) * A
    Sigma_tilde = diag_matrix(rep_vector(1.0, d)) - P1 * P1';
                           P = cholesky_decompose(Sigma_tilde);
                           Q = cholesky_decompose(Sigma_eta);
                           T = Q / P; // same as Q * inv(P)
                           Phi_mat = T * P1 * inverse(T);
                           return Phi_mat;
  }
  
  // Compute spectral density here
  complex_matrix compute_spec_dens(matrix Phi, matrix Sigma_eta, real freq) {
    complex_matrix [rows(Phi), rows(Phi)] spec_dens_X;
    complex_matrix [rows(Phi), rows(Phi)] spec_dens_Xi;
    complex_matrix [rows(Phi), rows(Phi)] Phi_op_inv;
    complex_matrix [rows(Phi), rows(Phi)] Phi_inv_H;
    complex_matrix [rows(Phi), rows(Phi)] spec_dens;
    complex_matrix [rows(Phi), rows(Phi)] Phi_comp = Phi;
    //tuple(complex_matrix, complex_vector) Phi_eigen;
    
    int d = rows(Phi);
    // Phi_op = diag_matrix(rep_vector(0, d)) - Phi_comp * exp(-1i * freq);
    // Phi_eigen = eigendecompose(Phi_op);
    // Phi_op_inv = Phi_eigen.1 * (1i./ Phi_eigen.2) * Phi_eigen.1;
    //Phi_op_inv = inverse(diag_matrix(rep_vector(0, d)) - Phi_comp * exp(-1i * freq));
    Phi_op_inv = diag_matrix(rep_vector(0, d)) / (diag_matrix(rep_vector(0, d)) - Phi_comp * exp(-1i * freq));
    Phi_inv_H = conj(Phi_op_inv');
    
    spec_dens_X = Phi_op_inv * Sigma_eta * Phi_inv_H;
    spec_dens_Xi = diag_matrix(rep_vector(pi()^2, d));
    spec_dens = spec_dens_X + spec_dens_Xi;
    
    return spec_dens;
  }

  real complex_wishart_lpdf (complex_matrix W, int v, complex_matrix Sigma) {
    complex log_likelihood;
    int r;
    //vector[] eigenvals_W;
    //vector[] eigenvals_Sigma;
    
    r = rows(W);
    complex_vector [rows(W)] eigenvals_W = eigenvalues_sym(W);
    complex_vector [rows(W)] eigenvals_Sigma = eigenvalues_sym(Sigma);
    
    complex sum_log_eig_W = 0;
    complex sum_log_eig_Sigma = 0;
    
    for (i in 1:r) {
      sum_log_eig_W += log(eigenvals_W[i]);
      sum_log_eig_Sigma += log(eigenvals_Sigma[i]);
    }
    
    log_likelihood = log(pi()^(v * (v - r)))
                    - logGamma_v(v) +
                    sum_log_eig_W - //sum(log(eigenvalues_sym(W))) -
                    v * sum_log_eig_Sigma - // v * sum(log(eigenvalues_sym(Sigma))) -
                    sum(diagonal(diag_matrix(rep_vector(0, r)) / Sigma * W));
    
    return get_real(log_likelihood);
    
  } // MAYBE TRY THIS WITH A REAL MATRIX AND SEE IF IT WORKS?
  
  real logGamma_v (int v) {
    real logG = 0;
    
    for (k in 1:v) {
      logG += lgamma(v - k + 1);
    }
    
    logG = logG + (v*(v-1)/2.0) * log(pi());
    return logG;
  }
  
  complex_matrix reconstruct_periodogr(matrix re_matrix, matrix im_matrix) {

    complex_matrix[rows(re_matrix), cols(re_matrix)] period_obs;

    for (i in 1:rows(re_matrix)) {
      for (j in 1:cols(re_matrix)) {
        period_obs[i,j] = to_complex(re_matrix[i,j], im_matrix[i,j]); 
      }
    }

    return period_obs;
  }

}

data {
  int d;    // dimension of the data at time t
  int<lower=0> nfreq;   // # time points (equally spaced)
  vector[nfreq] freqs;
  
  array[Q] matrix[N, J] re_matrices;
  array[Q] matrix[N, J] im_matrices;

  //int<lower=0> Tfin;   // time points (equally spaced)
  //matrix[d, Tfin] Y;
  //matrix[d, Tfin] X;
  
  vector[d*d] prior_mean_A;
  vector[d*d] diag_prior_var_A;
  vector[d] prior_mean_gamma;
  vector[d] diag_prior_var_gamma;
}

parameters {
  matrix[d, d] A;
  vector[d] gamma;
  // matrix[d, Tfin] X;
}

transformed parameters {
  matrix[d, d] Phi_mat;
  matrix[d, d] Sigma_eta_mat;
  
  Sigma_eta_mat = diag_matrix(exp(gamma));
  Phi_mat = to_VAR1_trans_mat(A, Sigma_eta_mat);
}

model {
  array[nfreq] complex_matrix[d, d] spec_dens;
  //array[nfreq] complex_matrix[d, d] periodogram;
  //complex_matrix[d,d] spec_dens[nfreq];
  
  to_vector(A) ~ normal(prior_mean_A, sqrt(diag_prior_var_A));
  gamma ~ normal(prior_mean_gamma, sqrt(diag_prior_var_gamma));
  // theta_phi ~ normal(0, sqrt(sigma_eta^2/(1 - phi^2)));
  // [get_real(theta_phi), get_imag(theta_phi)]' ~ normal(rep_vector(0, 2), rep_vector(sqrt(sigma_eta^2/(1 - phi^2)), 2));

// theta_sigma ~ normal(0, 1);


//for (k in 1:nfreq) {
  //  periodogram[k] = reconstruct_periodogr(re_matrices[k], re_matrices[k]);
  //  //print(periodogram[k]);
  //}

for (k in 1:nfreq) {
  spec_dens[k] = compute_spec_dens(Phi_mat, Sigma_eta_mat, freqs[k]);
  
  //periodogram[k] ~ complex_wishart(1, spec_dens[k]);
  print("loop iteration: ", k);
  // target += complex_wishart_lpdf(periodogram[k] | 1, spec_dens[k]);
}

  }
