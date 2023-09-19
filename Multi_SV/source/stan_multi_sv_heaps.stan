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
  
  matrix arima_stationary_cov2(matrix T, matrix R) {
    matrix[rows(T), cols(T)] Q0;
    matrix[rows(T) * rows(T), rows(T) * rows(T)] TT;
    vector[rows(T) * rows(T)] RR;
    int m;
    int m2;
    m = rows(T);
    m2 = m * m;
    RR = to_vector(tcrossprod(R));
    TT = kronecker_prod(T, T);
    // Q0 = to_matrix_colwise((diag_matrix(rep_vector(1.0, m2)) - TT) \ RR, m, m);
    //Q0 = to_matrix_colwise(rep_vector(1.0, m2), m, m);
    Q0 = diag_matrix(rep_vector(1.0, m));
    return Q0;
  }
  
  matrix arima_stationary_cov(matrix Phi, matrix Sigma_eta) {
    matrix[rows(Phi), cols(Phi)] Sigma1;
    matrix[rows(Phi) * rows(Phi), rows(Phi) * rows(Phi)] Phi_kron;
    vector[rows(Phi) * rows(Phi)] vec_Sigma1;
    int m;
    int m2;
    m = rows(Phi);
    m2 = m * m;
    Phi_kron = kronecker_prod(Phi, Phi);
    Sigma1 = to_matrix((diag_matrix(rep_vector(1.0, m2)) - Phi_kron) \ to_vector(Sigma_eta), m, m);
    
    return Sigma1;
  }
  
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
  
  /* Function to compute the matrix square root */
  matrix sqrtm(matrix A) {
    int m = rows(A);
    vector[m] root_root_evals = sqrt(sqrt(eigenvalues_sym(A)));
    matrix[m, m] evecs = eigenvectors_sym(A);
    matrix[m, m] eprod = diag_post_multiply(evecs, root_root_evals);
    return tcrossprod(eprod);
  }
  
  /* Function to transform A to P (inverse of part 2 of reparameterisation) */
  matrix AtoP(matrix A) {
    int m = rows(A);
    matrix[m, m] B = tcrossprod(A);
    for(i in 1:m) B[i, i] += 1.0;
    return mdivide_left_spd(sqrtm(B), A);
  }
  
  /* Function to perform the reverse mapping from the Appendix. The details of
     how to perform Step 1 are in Section S1.3 of the Supplementary Materials.
     Returned: a (2 x p) array of (m x m) matrices; the (1, i)-th component
               of the array is phi_i and the (2, i)-th component of the array
               is Gamma_{i-1}*/
  matrix[,] rev_mapping(matrix[] P, matrix Sigma) {
    int p = size(P);
    int m = rows(Sigma);
    matrix[m, m] phi_for[p, p];   matrix[m, m] phi_rev[p, p];
    matrix[m, m] Sigma_for[p+1];  matrix[m, m] Sigma_rev[p+1];
    matrix[m, m] S_for;           matrix[m, m] S_rev;
    matrix[m, m] S_for_list[p+1];
    matrix[m, m] Gamma_trans[p+1];
    matrix[m, m] phiGamma[2, p];
    // Step 1:
    Sigma_for[p+1] = Sigma;
    S_for_list[p+1] = sqrtm(Sigma);
    for(s in 1:p) {
      // In this block of code S_rev is B^{-1} and S_for is a working matrix
      S_for = - tcrossprod(P[p-s+1]);
      for(i in 1:m) S_for[i, i] += 1.0;
      S_rev = sqrtm(S_for);
      S_for_list[p-s+1] = mdivide_right_spd(mdivide_left_spd(S_rev, 
                              sqrtm(quad_form_sym(Sigma_for[p-s+2], S_rev))), S_rev);
      Sigma_for[p-s+1] = tcrossprod(S_for_list[p-s+1]);
    }
    // Step 2:
    Sigma_rev[1] = Sigma_for[1];
    Gamma_trans[1] = Sigma_for[1];
    for(s in 0:(p-1)) {
      S_for = S_for_list[s+1];
      S_rev = sqrtm(Sigma_rev[s+1]);
      phi_for[s+1, s+1] = mdivide_right_spd(S_for * P[s+1], S_rev);
      phi_rev[s+1, s+1] = mdivide_right_spd(S_rev * P[s+1]', S_for);
      Gamma_trans[s+2] = phi_for[s+1, s+1] * Sigma_rev[s+1];
      if(s>=1) {
        for(k in 1:s) {
          phi_for[s+1, k] = phi_for[s, k] - phi_for[s+1, s+1] * phi_rev[s, s-k+1];
          phi_rev[s+1, k] = phi_rev[s, k] - phi_rev[s+1, s+1] * phi_for[s, s-k+1];
        }
        for(k in 1:s) Gamma_trans[s+2] = Gamma_trans[s+2] + phi_for[s, k] * 
                                                               Gamma_trans[s+2-k];
      }
      Sigma_rev[s+2] = Sigma_rev[s+1] - quad_form_sym(Sigma_for[s+1], 
                                                      phi_rev[s+1, s+1]');
    }
    for(i in 1:p) phiGamma[1, i] = phi_for[p, i];
    for(i in 1:p) phiGamma[2, i] = Gamma_trans[i]';
    return phiGamma;
    // return phi_for[1, 1];
  }
}

data {
  int d; // dimension of the data at time t
  int p; // VAR(p)
  int<lower=0> Tfin;   // time points (equally spaced)
  matrix[d, Tfin] Y;
  //matrix[d, Tfin] X;
  
  //matrix[d, d] Sigma_1;
  vector[d*d] prior_mean_A;
  //matrix[d*d, d*d] prior_var_A;
  vector[d*d] diag_prior_var_A;
  vector[d] prior_mean_gamma;
  //matrix[d, d] prior_var_gamma;
  vector[d] diag_prior_var_gamma;
}

parameters {
  matrix[d, d] A;
  //real gamma_11;
  //real gamma_22;
  vector[d] gamma;
  matrix[d, Tfin] X;
}

transformed parameters { // define the mapping from A to Phi here
  matrix[d, d] Phi_mat;
  matrix[d, d] Sigma_eta_mat;
  matrix[d, d] P[1];
  Sigma_eta_mat = diag_matrix(exp(gamma));
  // Phi_mat = to_VAR1_trans_mat(A, Sigma_eta_mat);
  P[1] = AtoP(A);
  Phi_mat = rev_mapping(P, Sigma_eta_mat)[1,1];
}
model {
  //to_vector(A) ~ multi_normal(prior_mean_A, prior_var_A);
  //gamma ~ multi_normal(prior_mean_gamma, prior_var_gamma);
  
  to_vector(A) ~ normal(prior_mean_A, sqrt(diag_prior_var_A));
  gamma ~ normal(prior_mean_gamma, sqrt(diag_prior_var_gamma));
  
  // X[, 1] ~ multi_normal(rep_vector(0, d),
                           //                    arima_stationary_cov2(Phi_mat, cholesky_decompose(Sigma_eta_mat)));
  X[, 1] ~ multi_normal(rep_vector(0, d), arima_stationary_cov(Phi_mat, Sigma_eta_mat));
  // X[, 1] ~ multi_normal(rep_vector(0, d), diag_matrix(rep_vector(1.0, d)));
  
  
  for (t in 2:Tfin)
    X[, t] ~ normal(Phi_mat * X[, t-1], sqrt(exp(gamma)));
  //X[, t] ~ multi_normal(Phi_mat * X[, t-1], Sigma_eta_mat);
  for (t in 1:Tfin)
    Y[, t] ~ normal(rep_vector(0, d), exp(X[, t]/2));
}