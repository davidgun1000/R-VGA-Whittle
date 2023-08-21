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

## Forward mapping: Transform Phi -> P1

### 1. Compute autocovariances
test <- autocov_VAR1(Phi, Sigma_eta, 1)
Gamma_0 <- test[,,1]
Gamma_1 <- test[,,2]
# Gamma_0 <- chol2inv(chol(diag(2) - Phi %*% t(Phi))) %*% Sigma_eta
# Gamma_1 <- Phi %*% Gamma_0

### 2. 
Sigma_0 <- Sigma_0_star <- Gamma_0 #solve(diag(2) - Phi %*% t(Phi)) %*% Sigma_eta

S0 <- t(chol(Sigma_0))#sqrtm(Sigma_0)
S0_star <- S0

phi_11 <- t(Gamma_1) %*% chol2inv(chol(Gamma_0))
phi_11_star <- Gamma_1 %*% chol2inv(chol(Gamma_0))

# P1 <- chol2inv(chol(S0)) %*% phi_11 %*% S0_star
P1 <- t(solve(S0_star) %*% phi_11_star %*% S0)
## Reverse mapping: P1 -> Phi
### 1.
Sigma_1 <- Sigma_eta

#### Construct S_0
A <- diag(2) - P1 %*% t(P1)
B <- Sigma_1
C <- sqrtm(sqrtm(A) %*% B %*% sqrtm(A))
# 
sqrtA_inv <- solve(sqrtm(A))
S0_rev2 <- sqrtA_inv %*% C %*% sqrtA_inv

S0_rev <- solve(A) %*% sqrtm(A %*% B) #+ (diag(2) - chol2inv(chol(A)) %*%)
# Check: S0_rev %*% (diag(2) - P1 %*% t(P1)) %*% t(S0_rev)
Sigma_0_rev <- S0_rev %*% t(S0_rev)
browser()


# S0_rev <- sqrtm(Sigma_0)
Gamma_0_rev <- Sigma_0_rev

### 2.
# Sigma_0_rev <- Sigma_0_star_rev <- Gamma_0_rev

# S0_rev <- t(chol(Sigma_0_rev))
S0_star_rev <- S0_rev

phi_11_rev <- S0_rev %*% P1 %*% solve(S0_star_rev)
phi_11_star_rev <- S0_star_rev %*% t(P1) %*% solve(S0_rev)
