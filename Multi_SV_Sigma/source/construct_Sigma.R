index_to_i_j_rowwise_nodiag <- function(k) { # maps vector entries to lower triangular indices
  kp <- k - 1
  p  <- floor((sqrt(1 + 8 * kp) - 1) / 2)
  i  <- p + 2
  j  <- kp - p * (p + 1) / 2 + 1
  c(i, j)
}

construct_Sigma_eta <- function(theta, d, use_chol = T) { #d is the dimension of Sigma_eta
  nlower <- d*(d-1)/2
  L <- diag(exp(theta[(d+1):(2*d)]))
  offdiags <- theta[-(1:(2*d))] # off diagonal elements are those after the first 2*d elements
  
  if (use_chol) {
    for (k in 1:nlower) {
      ind <- index_to_i_j_rowwise_nodiag(k)
      L[ind[1], ind[2]] <- offdiags[k]
    }
    Sigma_eta <- L %*% t(L)
  } else {
    Sigma_eta <- L
  }
  return(Sigma_eta)
}

to_triangular <- function (x, d) { ## for stan
  # // could check rows(y) = K * (K + 1) / 2
  # matrix[K, K] y;    
  K = d-1
  pos = 1
  nlower = d*(d-1)/2
  # L = matrix(NA, d, d)
  L = exp(diag(x[1:d]))
  for (i in 2:d) {
    for (j in 1:(i-1)) {
      L[i,j] = x[pos]
      pos = pos + 1
    }
  }
  
  return (L)
  # for (int i = 1; i < K; ++i) {
  #   for (int j = 1; j <= i; ++j) {
  #     y[i, j] = y_basis[pos];
  #     pos += 1;
  #   }
  # }
  # return y;
}
