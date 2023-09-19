## Bivariate SV model

setwd("~/R-VGA-Whittle/Multi_SV/")

source("./source/compute_whittle_likelihood_var1.R")

library("mvtnorm")
library("astsa")

a11 <- 0.9
a12 <- 0
a21 <- 0
a22 <- 0.7

A <- matrix(c(a11, a12, a21, a22), 2, 2, byrow = T)

sigma_eta1 <- 0.2
sigma_eta2 <- 0.6
Sigma_eta <- diag(c(sigma_eta1, sigma_eta2))

# sigma_eps1 <- 0.01
# sigma_eps2 <- 0.02
# 
# Sigma_eps <- diag(c(sigma_eps1, sigma_eps2))

Tfin <- 1000
x1 <- c(1, 1)
X <- matrix(NA, nrow = length(x1), ncol = Tfin+1) # x_0:T
X[, 1] <- x1
# Y <- matrix(NA, nrow = length(x1), ncol = Tfin) # y_1:T
# Y[, 1] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
for (t in 1:Tfin) {
  X[, t+1] <- A %*% X[, t] + t(rmvnorm(1, c(0, 0), Sigma_eta))
  # V <- diag(exp(X[, t+1])/2)
  # Y[, t] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
}

par(mfrow = c(2,1))
plot(X[1, ], type = "l")
plot(X[2, ], type = "l")

# plot(Y[1, ], type = "l")
# plot(Y[2, ], type = "l")

## Calculate Whittle likelihood

a_grid <- seq(0.1, 1, length.out = 50)
sigma_eta_grid <- seq(0.1, 2, length.out = 50)

# test <- compute_whittle_likelihood_multi_sv(Y = Y, 
#                                                params = list(A = A, Sigma_eta = Sigma_eta))
# browser()

llh <- c()

options(warn=2)
for (j in 1:length(a_grid)) {
  cat("j = ", j, "\n")
  # Amat <- matrix(c(a_grid[j], a12, a21, a22), 2, 2, byrow = T)
  Sigma_eta_j <- diag(c(sigma_eta_grid[j], sigma_eta2))
  llh[j] <- compute_whittle_likelihood_var1(X = X, 
                                            params = list(A = A, #Amat, 
                                                          Sigma_eta = Sigma_eta_j))
}

if (any(Im(llh) - 0 > 1e-10)) { # if imaginary part is effectively zero, get rid of it
  print("Warning: Imaginary part of the log likelihood is non-zero")
} else {
  llh <- Re(llh) 
}

# plot(a_grid, llh, type = "l")
# abline(v = a11, col = "red", lty = 2)

plot(sigma_eta_grid, llh, type = "l")
abline(v = sigma_eta1, col = "red", lty = 2)

# ## Fourier frequencies
# k <- seq(-ceiling(Tfin/2)+1, floor(Tfin/2), 1)
# k_in_likelihood <- k[k >= 1 & k <= floor((Tfin-1)/2)]
# freq <- 2 * pi * k_in_likelihood / Tfin
# 
# Y_tilde <- log(Y^2) - rowMeans(log(Y^2))
# 
# # J <- fft(Y)
# # J <- t(J)
# # I_omega <- 1/(2*pi*Tfin) * J %*% Conj(t(J)) # Should this be TxT?
# 
# # J_test <- 0
# # for (t in 1:Tfin) {
# #   J_test <- J_test + Y[, t] * exp(-1i * freq[1] * t) 
# # }
# # test <- exp(-1i * freq[1] * (0:(Tfin-1)))
# # test2 <- t(cbind(test, test))
# # test2 <- t(cbind(test, test))
# # J_all <- Y * test2 # but there is no Y_0???
# # J_test2 <- rowSums(J_all)
# 
# J <- list()
# for (k in 1:length(freq)) {
#   test <- exp(-1i * freq[k] * (0:(Tfin-1)))
#   # test2 <- t(cbind(test, test))
#   # J_manual <- X[, 1] * exp(-1i * freq[1] * 1) # need to check the fft calculation manually
#   J_all <- Y_tilde * t(cbind(test, test)) # but there is no Y_0???
#   J[[k]] <- rowSums(J_all)
# }
# 
# I_all <- lapply(J, function(M, Tfin) 1/Tfin * M %*% Conj(t(M)), Tfin = Tfin)
# 
# # Spectral density matrix
# Phi_0 <- diag(2)
# Phi_1 <- A
# Theta <- diag(2)
# 
# log_likelihood <- 0
# for (k in 1:length(freq)) {
#   Phi_inv <- solve(Phi_0 + Phi_1 * exp(-1i * freq[k]))
#   Phi_inv_H <- Conj(t(Phi_inv))
#   
#   spec_dens_X <-1/(2*pi) * Phi_inv %*% Theta %*% Sigma_eta %*% Theta %*% Phi_inv_H  
#   
#   spec_dens_Xi <- diag(pi^2/2, 2)
#   
#   spec_dens <- spec_dens_X + spec_dens_Xi  
#   
#   part2 <- sum(diag(solve(spec_dens) %*% I_all[[k]]))
#   
#   # log(det(spec_dens))
#   
#   det_spec_dens <- prod(eigen(spec_dens)$values)
#   part1 <- log(det_spec_dens)
#   
#   log_likelihood <- log_likelihood - (part1 + part2)
#   # test <- spec_dens[1, 2] * spec_dens[2, 1] - spec_dens[1, 1] * spec_dens[2, 2]
#   
# }
# 
