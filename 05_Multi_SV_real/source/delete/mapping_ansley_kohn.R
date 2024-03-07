library(expm)
library("stcos")

phi11 <- 0.8
phi12 <- 0.05
phi21 <- 0.2
phi22 <- 0.9

Phi <- matrix(c(phi11, phi12, phi21, phi22), 2, 2, byrow = T)

sigma_eta1 <- 0.2
sigma_eta2 <- 0.6
Sigma_eta <- diag(c(sigma_eta1, sigma_eta2))

## Forward map
### 1. Map original parameter space to partial autocovariances
autocovs <- autocov_VAR1(Phi, Sigma_eta, 1)
Gamma_0 <- autocovs[,,1]
Gamma_1 <- autocovs[,,2]

L0 <- t(chol(Gamma_0))
L0_star <- L0
P1 <- solve(L0) %*% Gamma_1 %*% solve(t(L0_star))

### 2. Map partial autocov to unconstrained space
B_inv <- t(chol(diag(2) - P1 %*% t(P1)))
A1 <- solve(B_inv) %*% P1

###################################
###       Reverse mapping       ###
###################################

### 1. Map from unconstrained space to partial autocovs
B1_rev <- t(chol(diag(2) + A1 %*% t(A1)))
P1_rev <- solve(B1_rev) %*% A1

### 2. Map from partial autocovs to original parameters 
Gamma_0_rev <- diag(2) #Sigma_eta
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
