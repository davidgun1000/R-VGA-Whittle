functions {
  /* Function to compute the matrix square root */
  matrix sqrtm(matrix A) {
    int m = rows(A);
    vector[m] root_root_evals = sqrt(sqrt(eigenvalues_sym(A)));
    matrix[m, m] evecs = eigenvectors_sym(A);
    matrix[m, m] eprod = diag_post_multiply(evecs, root_root_evals);
    return tcrossprod(eprod);
  }
  /* Function to map unconstrained C to positive definite V and orthogonal Q. The details
     of the mapping are given in Section S3 of the Supplementary Materials.
     Returned: a 2-dimensional array of (m x m) matrices; the 1st component
               of the array is V and the 2nd component is Q. */
  matrix[] CtoVQ(matrix C) {
    int m = rows(C);
    matrix[m, m] Dsq = diag_matrix(singular_values(C).^2);
    matrix[m, m] U = svd_U(C);
    matrix[m, m] V = svd_V(C);
    matrix[m, m] VQ[2];
    VQ[1] = V * Dsq * V'; // V
    VQ[2] = V * U'; // Q
    return VQ;
  }
  /* Function to perform Algorithm [VQ] from Roy et al. (2019).
     Returned: a (2 x p) array of (m x m) matrices; the (1, i)-th component
               of the array is phi_i and the (2, i)-th component of the array
               is Gamma_{i-1} (assuming M = Sigma) */
  matrix[,] rev_mapping(matrix[] V, matrix[] Q, matrix M) {
    int p = size(V);
    int m = rows(M);
    matrix[m, m] U[p+1];
    matrix[m*(p+1), m*(p+1)] Uunder[p+1];
    matrix[m*p, m] eps[p];
    matrix[m*p, m] kappa[p];
    matrix[m, m] D[p+1];
    matrix[m, p*m] A;
    matrix[m, m] phiGamma[2, p];
    U[1] = M;
    for(i in 1:p) {
      U[1] = U[1] + V[i];
    } // U(0)
    Uunder[1][1:m, 1:m] = U[1];
    D[1] = U[1];
    U[2] = sqrtm(V[1]) * Q[1] * sqrtm(U[1]); // U(1), etc.
    eps[1][1:m,] = U[2]';
    kappa[1][1:m,] = U[2];
    D[2] = U[1] - kappa[1][1:m,]' * mdivide_left_spd(Uunder[1][1:m, 1:m], kappa[1][1:m,]);
    Uunder[2][1:m, 1:m] = U[1]; Uunder[2][1:m, (m+1):(2*m)] = eps[1][1:m,]';
    Uunder[2][(m+1):(2*m), 1:m] = eps[1][1:m,]; Uunder[2][(m+1):(2*m), (m+1):(2*m)] = Uunder[1][1:m, 1:m];
    if(p > 1) {
      for(i in 2:p) {
        int end = (i-1)*m;
        U[i+1] = eps[i-1][1:end,]' * mdivide_left_spd(Uunder[i-1][1:end, 1:end], kappa[i-1][1:end,]) +
                     sqrtm(V[i]) * Q[i] * sqrtm(D[i]);
        eps[i][1:end,] = eps[i-1][1:end,];
        eps[i][(end+1):(end+m),] = U[i+1]';
        kappa[i][1:m,] = U[i+1];
        kappa[i][(m+1):(end+m),] = kappa[i-1][1:end,];
        D[i+1] = U[1] - kappa[i][1:(end+m),]' * mdivide_left_spd(Uunder[i][1:(end+m), 1:(end+m)], kappa[i][1:(end+m),]);
        Uunder[i+1][1:m, 1:m] = U[1]; Uunder[i+1][1:m, (m+1):(end+2*m)] = eps[i][1:(end+m),]';
        Uunder[i+1][(m+1):(end+2*m), 1:m] = eps[i][1:(end+m),]; Uunder[i+1][(m+1):(end+2*m), (m+1):(end+2*m)] = Uunder[i][1:(end+m), 1:(end+m)];
      }
    }
    A = mdivide_right_spd(eps[p]', Uunder[p][1:(p*m), 1:(p*m)]);
    for(i in 1:p) phiGamma[1, i] = A[,((i-1)*m+1):(i*m)];
    for(i in 1:p) phiGamma[2, i] = Uunder[p][1:m,((i-1)*m+1):(i*m)]';
    return phiGamma;
  }
}
data {
  int<lower=1> m; // Dimension of observation vector
  int<lower=1> p; // Order of VAR model
  int<lower=1> N; // Length of time series
  vector[m] y[N]; // Time series
  // Hyperparameters in inverse Wishart prior for Sigma (the scale matrix is the identity)
  real<lower=m+3> df; // Degrees of freedom (limit ensures finite variance)
}
transformed data {
  vector[p*m] y_1top;                // y_1, ..., y_p
  vector[m] mu = rep_vector(0.0, m); // (Zero)-mean of VAR process
  matrix[m, m] identity;             // Identity matrix
  for(t in 1:p) y1top[((t-1)*m+1):(t*m)] = y[t];
  identity = diag_matrix(rep_vector(1.0, m));
}
parameters {
  matrix[m, m] C[p];   // The C_i
  cov_matrix[m] Sigma; // Error variance, Sigma
}
transformed parameters {
  matrix[m, m] phi[p];   // The phi_i
  cov_matrix[p*m] Gamma; // (Stationary) variance of (y_1, ..., y_p)
  {
    matrix[m, m] VQ[2];
    matrix[m, m] V[p];
    matrix[m, m] Q[p];
    matrix[m, m] phiGamma[2, p];
    for(i in 1:p) {
      VQ = CtoVQ(C[i]);
      V[i] = VQ[1]; Q[i] = VQ[2];
    }
    phiGamma = rev_mapping(V, Q, Sigma);
    phi = phiGamma[1];
    for(i in 1:p) {
      for(j in 1:p) {
        if(i<=j) Gamma[((i-1)*m+1):(i*m), ((j-1)*m+1):(j*m)] = phiGamma[2, j-i+1];
        else Gamma[((i-1)*m+1):(i*m), ((j-1)*m+1):(j*m)] = phiGamma[2, i-j+1]';
      }
    }
  }
}
model {
  vector[p*m] mut_init;    // Marginal mean of (y_1^T, ..., y_p^T)^T
  vector[m] mut_rest[N-p]; // Conditional means of y_{p+1}, ..., y_{N}
  // Likelihood:
  for(t in 1:p) mut_init[((t-1)*m+1):(t*m)] = mu;
  for(t in (p+1):N) {
    mut_rest[t-p] = mu;
    for(i in 1:p) {
      mut_rest[t-p] += phi[i] * (y[t-i] - mu);
    }
  }
  y1top ~ multi_normal(mut_init, Gamma);
  y[(p+1):N] ~  multi_normal(mut_rest, Sigma);
  // Prior:
  Sigma ~ inv_wishart(df, identity);
  for(i in 1:p) to_vector(C[i]) ~ std_normal();
}
