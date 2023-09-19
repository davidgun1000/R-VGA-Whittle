library(expm)
library("stcos")

phi11 <- 0.9
phi12 <- 0.1
phi21 <- 0.2
phi22 <- 0.7

Phi <- matrix(c(phi11, phi12, phi21, phi22), 2, 2, byrow = T)

sigma_eta1 <- 0.2
sigma_eta2 <- 0.6
Sigma_eta <- diag(c(sigma_eta1, sigma_eta2))

sqrt_mat <- function(A) {
  m <- nrow(A)
  eig <- eigen(A)
  eigvals <- eig$values
  eigvecs <- eig$vectors
  sqrt_sqrt_D <- diag(sqrt(sqrt(eigvals)))
  sqrtA <- tcrossprod(eigvecs %*% sqrt_sqrt_D)
  return (sqrtA)
}


## Forward mapping: Transform Phi -> P1

forward_map <- function(Phi, Sigma_eta) {
  ## 1. Phi to P1
  autocovs <- autocov_VAR1(Phi, Sigma_eta, 1)
  Gamma_0 <- autocovs[,,1]
  Gamma_1 <- autocovs[,,2]
  
  Sigma_0 <- Gamma_0
  S0 <- sqrt_mat(Sigma_0)
  phi_11 <- t(Gamma_1) %*% solve(Gamma_0)
  P1 <- solve(S0) %*% phi_11 %*% S0
  
  ## 2. P1 to A
  B_inv <- sqrt_mat(diag(nrow(P1)) - tcrossprod(P1))
  A <- solve(B_inv) %*% P1
  
  return(list(A = A, P1 = P1, Gamma_0 = Gamma_0, S0 = S0))
}

backward_map <- function(A, Sigma_eta) {
  
  ### 0. A to P1
  B_rev = sqrt_mat(diag(nrow(A)) + tcrossprod(A))
  P1_rev = solve(B_rev) %*% A
  
  ### 1. (Sigma, P1) to Gamma_0
  Sigma_1_rev <- Sigma_eta
  # S1 <- sqrt_mat(Sigma_1)
  U <- diag(nrow(Sigma_1_rev)) - tcrossprod(P1_rev)
  V <- Sigma_1_rev
  sqrtU <- sqrtm(U)
  inv_sqrtU <- solve(sqrtU)
  C <- sqrtm(sqrtm(U) %*% V %*% sqrtm(U))
  S0_rev <- inv_sqrtU %*% C %*% inv_sqrtU
  # S0_rev <- t(chol(V)) %*% solve(t(chol(U)))
  # S0_rev <- sqrt_mat(V) %*% solve(sqrt_mat(U))
  
  Sigma_0_rev <- tcrossprod(S0_rev)
  Gamma_0_rev <- Sigma_0_rev
  
  ### 2. (P1, Gamma_0) to Phi
  Phi <- S0_rev %*% P1_rev %*% solve(S0_rev)
  
  return(list(Phi = Phi, P1 = P1_rev, S0 = S0_rev))
}

forward_out <- forward_map(Phi, Sigma_eta)
A <- forward_out$A

backward_out <- backward_map(A, Sigma_eta)
Phi_test <- backward_out$Phi

