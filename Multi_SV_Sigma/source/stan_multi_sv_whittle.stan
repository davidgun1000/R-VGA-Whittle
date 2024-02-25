functions {

    // Compute spectral density here
  complex_matrix compute_spec_dens(matrix Phi, complex_matrix Sigma_eta, real freq) {
    complex_matrix [rows(Phi), rows(Phi)] spec_dens_X;
    complex_matrix [rows(Phi), rows(Phi)] spec_dens_Xi;
    complex_matrix [rows(Phi), rows(Phi)] Phi_op;
    complex_matrix [rows(Phi), rows(Phi)] Phi_op_inv;
    
    complex_matrix [rows(Phi), rows(Phi)] Phi_inv_H;
    complex_matrix [rows(Phi), rows(Phi)] spec_dens;
    // complex_matrix [rows(Phi), rows(Phi)] Phi_comp = Phi;
    int d = rows(Phi);
    
    // Phi_op_inv = diag_matrix(rep_vector(1, d));// / (diag_matrix(rep_vector(1, d)) - Phi * exp(-1i * freq));
    // print("Phi_op_inv =", Phi_op_inv);
    // Phi_inv_H = conj(Phi_op_inv');
    
    Phi_op = diag_matrix(rep_vector(1, d)) - Phi * exp(-1i * freq);
    Phi_op_inv = diag_matrix(1/diagonal(Phi_op));
    Phi_inv_H = conj(Phi_op_inv');
    //print("Phi_op = ", Phi_op);
    //print("Phi_op_inv = ", Phi_op_inv);
    
    spec_dens_X = Phi_op_inv * Sigma_eta * Phi_inv_H;

    spec_dens_Xi = diag_matrix(rep_vector(pi()^2/2, d));
    spec_dens = spec_dens_X + spec_dens_Xi;
    // print("spec_dens_X =", spec_dens_X);
    return spec_dens;
  }

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

  matrix to_lowertri(vector x, int d) {
    int pos = 1;
    matrix[d, d] L;
    vector[num_elements(x) - d] lower_entries;
    
    L = diag_matrix(exp(x[1:d])); // the first d elements form the diagonal entries
    lower_entries = x[(d+1):num_elements(x)];
    
    for (i in 2:d) {
      for (j in 1:(i-1)) {
        L[i, j] = lower_entries[pos];
        pos += 1;
      }
    }
    
    return L;
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

  complex_vector log_comp_vec (complex_vector v) {
    complex_vector [size(v)] log_v;
    for (i in 1:size(v)) {
      log_v[i] = log(v[i]);
    }
    return log_v;
  }

  real complex_wishart_lpdf (complex_matrix W, int v, complex_matrix Sigma) {
    complex log_likelihood = 0;
    complex log_likelihood1 = 0;
    complex log_likelihood2 = 0;
    
    int r = rows(W);
    
    complex sum_log_eig_W = 0;
    complex sum_log_eig_Sigma = 0;
    
    for (i in 1:r) {
      // print("eigenvalues_W =", eigenvalues(W)[i]);
      // print("log eigenvalues_W =", log(eigenvalues(W)[i]));
      
      sum_log_eig_W += log(eigenvalues(W)[i]);
      sum_log_eig_Sigma += log(eigenvalues(Sigma)[i]);
    }
    
    // print("sum_log_eig_W =", sum_log_eig_W);
    // print("sum_log_eig_Sigma =", sum_log_eig_Sigma);
    
    // log_likelihood = (v * (v - r)) * log(pi())
    //                 - logGamma(v, v) +
    //                 sum_log_eig_W - //sum(log(eigenvalues_sym(W))) -
    //                 v * sum_log_eig_Sigma - // v * sum(log(eigenvalues_sym(Sigma))) -
    //                 sum(diagonal(diag_matrix(rep_vector(1, r)) / Sigma * W));
    log_likelihood1 = sum_log_eig_Sigma; 
    // log_likelihood2 = trace((diag_matrix(rep_vector(1, r)) / Sigma) * W);
    log_likelihood2 = trace(W / Sigma);
    print("log_likelihood1 =", log_likelihood1);
    print("log_likelihood2 =", log_likelihood2);
    print("W / Sigma = ", W / Sigma);
    log_likelihood = -(log_likelihood1 + log_likelihood2); 
    // print("log_likelihood =", log_likelihood);
    
    
    return get_real(log_likelihood);
    
  } // MAYBE TRY THIS WITH A REAL MATRIX AND SEE IF IT WORKS?
  
  real complex_wishart2_lpdf (array[] complex_matrix W, int v, array[] complex_matrix Sigma) {
    real log_likelihood = 0;
    complex log_likelihood_k = 0;
    // complex log_likelihood2 = 0;
    complex sum_log_eig_Sigma = 0;
    int r = rows(W[1]);
    int nfreq = size(W);
    
    for (k in 1:nfreq) {
      sum_log_eig_Sigma = 0;
      for (i in 1:r) { # compute log det(Sigma)
        sum_log_eig_Sigma += log(eigenvalues(Sigma[k])[i]);
      }
      
      log_likelihood_k = -(sum_log_eig_Sigma + trace(W[k] / Sigma[k]));
    
      log_likelihood += get_real(log_likelihood_k); 
      
    }
    //print("log_likelihood =", log_likelihood);
    // log_likelihood = get_real(log_likelihood);
    return log_likelihood;
    
  } // MAYBE TRY THIS WITH A REAL MATRIX AND SEE IF IT WORKS?
  
  
  // real complex_wishart_test (complex_matrix W, int v, complex_matrix Sigma) {
  //   complex log_likelihood = 0;
  //   complex log_likelihood1 = 0;
  //   complex log_likelihood2 = 0;
  //   
  //   // complex_matrix[rows(W), cols(W)] log_likelihood2;
  //   
  //   int r = rows(W);
  //   
  //   complex sum_log_eig_W = 0;
  //   complex sum_log_eig_Sigma = 0;
  //   
  //   for (i in 1:r) {
  //     // print("eigenvalues_W =", eigenvalues(W)[i]);
  //     // print("log eigenvalues_W =", log(eigenvalues(W)[i]));
  //     
  //     sum_log_eig_W += log(eigenvalues(W)[i]);
  //     sum_log_eig_Sigma += log(eigenvalues(Sigma)[i]);
  //   }
  //   
  //   log_likelihood1 = sum_log_eig_Sigma; 
  //   // log_likelihood2 = trace((diag_matrix(rep_vector(1, r)) / Sigma) * W);
  //   log_likelihood2 = trace(W / Sigma);
  //   
  //   log_likelihood = -(log_likelihood1 + log_likelihood2);                
  //   
  //   return get_real(log_likelihood);
  //   
  // } // MAYBE TRY THIS WITH A REAL MATRIX AND SEE IF IT WORKS?
  // 
  
  real logGamma (int a, int n) {
    real logG = 0;
    
    for (k in 1:n) {
      logG += lgamma(a - k + 1);
    }
    
    logG = logG + (n*(n-1)/2.0) * log(pi());
    return logG;
  }

}

data {
  int d;    // dimension of the data at time t
  int<lower=0> nfreq;   // # time points (equally spaced)
  vector[nfreq] freqs;
  
  array[nfreq] matrix[d, d] re_matrices;
  array[nfreq] matrix[d, d] im_matrices;

  //int<lower=0> Tfin;   // time points (equally spaced)
  //matrix[d, Tfin] Y;
  //matrix[d, Tfin] X;

  vector[d] prior_mean_Phi;
  vector[d] diag_prior_var_Phi;
  vector[d + d*(d-1)/2] prior_mean_gamma;
  vector[d + d*(d-1)/2] diag_prior_var_gamma;

  int transform; # indicator for using logit or arctanh transform
  
  //  delete later:
  matrix[d,d] truePhi;
  matrix[d,d] trueSigma;
}

parameters {
  vector[d] theta_phi;
  vector[d + d*(d-1)/2] gamma;
  //matrix[Tfin, d] X;
}

transformed parameters {
  matrix[d, d] Phi_mat;
  cov_matrix[d] Sigma_eta_mat;
  matrix[d, d] L;
  
  L = to_lowertri(gamma, d);
  Sigma_eta_mat = L*L';
  // print("Sigma_eta = ", Sigma_eta_mat);
  
  Phi_mat = transform ? diag_matrix(tanh(theta_phi[1:d])) : diag_matrix(1 / (1+exp(-theta_phi[1:d])));
}

model {
  array[nfreq] complex_matrix[d, d] spec_dens;
  array[nfreq] complex_matrix[d, d] periodogram;
  //complex_matrix[d,d] spec_dens[nfreq];
  
  theta_phi ~ normal(prior_mean_Phi, sqrt(diag_prior_var_Phi));
  gamma ~ normal(prior_mean_gamma, sqrt(diag_prior_var_gamma));
  
  //real llh_stan = 0;
  //for (k in 1:1) {
  //  periodogram[k] = reconstruct_periodogr(re_matrices[k], im_matrices[k]);
  //  print("periodogram = ", periodogram[k]);
  //  spec_dens[k] = compute_spec_dens(truePhi, trueSigma, freqs[k]);
  // // print("spec_dens = ", spec_dens[k]);
  //  periodogram[k] ~ complex_wishart(1, spec_dens[k]);
  //  // target += complex_wishart(periodogram[k], 1, spec_dens[k]);
  //  // print("llh_stan = ", llh_stan);
  //}
  
  for (k in 1:nfreq) {
     spec_dens[k] = compute_spec_dens(Phi_mat, Sigma_eta_mat, freqs[k]);
    // print("spec_dens = ", spec_dens[k]);
    periodogram[k] = reconstruct_periodogr(re_matrices[k], im_matrices[k]);
    // print("periodogram = ", periodogram[k]);
    // periodogram[k] ~ complex_wishart(1, spec_dens[k]);
    // target += complex_wishart_lpdf(periodogram[k] | 1, spec_dens[k]);
  
  }
  
  periodogram ~ complex_wishart2(1, spec_dens);
  // target += complex_wishart2_lpdf(periodogram | 1, spec_dens);
}

