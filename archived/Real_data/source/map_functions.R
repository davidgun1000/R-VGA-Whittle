forward_map <- function(Phi, Sigma_eta) { ## map from VAR(1) parameters to unconstrained space
  d <- dim(Phi)[1]
  autocovs <- autocov_VAR1(Phi, Sigma_eta, 1)
  Gamma_0 <- autocovs[,,1]
  Gamma_1 <- autocovs[,,2]
  
  L0 <- t(chol(Gamma_0))
  L0_star <- L0
  P1 <- solve(L0) %*% Gamma_1 %*% solve(t(L0_star))
  
  ### 2. Map partial autocov to unconstrained space
  B_inv <- t(chol(diag(d) - P1 %*% t(P1)))
  A1 <- solve(B_inv) %*% P1
  
  return(A1)
}

backward_map <- function(A, Sigma_eta) { # map from unconstrained space back to VAR(1) parameters 
  d <- dim(A)[1]
  ### 1. Map from unconstrained space to partial autocovs
  B1_rev <- t(chol(diag(d) + A %*% t(A)))
  P1_rev <- solve(B1_rev) %*% A
  
  ### 2. Map from partial autocovs to original parameters 
  Gamma_0_rev <- diag(d) #Sigma_eta
  Sigma_0_rev <- Gamma_0_rev
  L0_rev <- t(chol(Gamma_0_rev))
  L0_star_rev <- L0_rev
  
  phi_tilde <- L0_rev %*% P1_rev %*% solve(L0_star_rev)
  # phi_star_tilde <- L0_star_rev %*% t(P1_rev) %*% solve(L0_rev)
  
  Sigma_1_rev <- Sigma_0_rev - phi_tilde %*% Sigma_0_rev %*% t(phi_tilde)
  
  ## To solve equation X*A*t(X) = B, let A = P*t(P) and B = Q*t(Q)
  ## then solve for XP = QU, where U can be any orthogonal matrix
  ## here we choose U = identity matrix
  P <- t(chol(Sigma_1_rev))
  Q <- t(chol(Sigma_eta))
  X <- Q %*% solve(P)
  
  Phi_rev <- X %*% phi_tilde %*% solve(X)
  
  return(Phi_rev)
}

backward_map2 <- function(A, Sigma_eta) { ## to test the transform implemented in Stan
  d = dim(A)[1]
  B = t(chol(diag(d) + A %*% t(A)))
  P1 = solve(B) %*% A
  Sigma_tilde = diag(d) - P1 %*% t(P1)
  P = t(chol(Sigma_tilde))
  Q = t(chol(Sigma_eta))
  Tmat = Q %*% solve(P)
  Phi = Tmat %*% P1 %*% solve(Tmat)
}

backward_map_tf <- function(A, Sigma_eta) { # map from unconstrained space back to VAR(1) parameters 
  
  S <- as.integer(dim(A)[1])
  d <- as.integer(dim(A)[2])
  # A_test <- as.matrix(A[1,,])
  # Sigma_eta_test <- as.matrix(Sigma_eta[1,,])
  
  ### 1. Map from unconstrained space to partial autocovs
  # B1_rev <- t(chol(diag(2) + A_test %*% t(A_test)))
  # P1_rev <- solve(B1_rev) %*% A_test
  
  Id_tf <- tf$cast(tf$eye(d), dtype = "float64")
  Id_tf <- tf$reshape(Id_tf, c(1L, dim(Id_tf)))
  Id_tiled <- tf$tile(Id_tf, c(S, 1L, 1L))
  
  B1_rev_tf <- tf$linalg$cholesky(Id_tiled + tf$linalg$matmul(A, tf$transpose(A, perm = c(0L, 2L, 1L))))
  P1_rev_tf <- tf$linalg$matmul(tf$linalg$inv(B1_rev_tf), A)
  
  ### 2. Map from partial autocovs to original parameters 
  # Sigma_0_rev <- Gamma_0_rev <- diag(2)
  # L0_rev <- t(chol(Gamma_0_rev))
  # L0_star_rev <- L0_rev
  
  # Gamma_0_rev_tf <- tf$eye(2L)
  # Gamma_0_reshape <- tf$reshape(Gamma_0_tf, c(1L, dim(Gamma_0_tf)))
  # Gamma_0_tiled <- tf$tile(Phi_0_reshape, c(S, 1L, 1L))
  # 
  # Sigma_0_rev_tf <- Gamma_0_tiled
  Sigma_0_rev_tf <- Id_tiled
  L0_rev_tf <- tf$linalg$cholesky(Sigma_0_rev_tf)
  
  # phi_tilde <- L0_rev %*% P1_rev %*% solve(L0_star_rev)
  phi_tilde_tf <- tf$linalg$matmul(tf$linalg$matmul(L0_rev_tf, P1_rev_tf), L0_rev_tf)
  
  # Sigma_1_rev <- Sigma_0_rev - phi_tilde %*% Sigma_0_rev %*% t(phi_tilde)
  Sigma_1_rev_tf <- Sigma_0_rev_tf - tf$linalg$matmul(tf$linalg$matmul(phi_tilde_tf, Sigma_0_rev_tf),
                                                      tf$transpose(phi_tilde_tf, perm = c(0L, 2L, 1L)))
  
  ## To solve equation X*A*t(X) = B, let A = P*t(P) and B = Q*t(Q)
  ## then solve for XP = QU, where U can be any orthogonal matrix
  ## here we choose U = identity matrix
  # P <- t(chol(Sigma_1_rev))
  # Q <- t(chol(Sigma_eta))
  # X <- Q %*% solve(P)
  
  P_tf <- tf$linalg$cholesky(Sigma_1_rev_tf)
  Q_tf <- tf$linalg$cholesky(Sigma_eta)
  X_tf <- tf$linalg$matmul(Q_tf, tf$linalg$inv(P_tf)) 
  
  # Phi_rev <- X %*% phi_tilde %*% solve(X)
  
  Phi_rev_tf <- tf$linalg$matmul(tf$linalg$matmul(X_tf, phi_tilde_tf), tf$linalg$inv(X_tf))
  
  return(Phi_rev_tf)
}

## Function to compute autocovariance matrix for VAR(1)
compute_autocov_VAR1 <- function(Phi, Sigma_eta) {
  d <- dim(Phi)[1]
  Phi_kron <- kronecker(Phi, Phi)
  vec_Sigma1 <- solve(diag(ncol(Phi_kron)) - Phi_kron) %*% c(Sigma_eta) 
  Sigma1 <- matrix(vec_Sigma1, d, d)
  return(Sigma1)
}
