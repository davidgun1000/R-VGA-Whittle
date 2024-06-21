## Test DFT ##

rm(list = ls())

setwd("~/R-VGA-Whittle/Multi_SV/")

library(mvtnorm)
library(astsa)
library(beyondWhittle)

source("./source/compute_whittle_likelihood_sv.R")
source("./source/compute_whittle_likelihood_multi_sv.R")

#######################
##   Generate data   ##
#######################

phi11 <- 0.5 #0.9
phi12 <- 0.3 #0.1 # dataset1: 0.1, dataset2 : -0.5
phi21 <- 0.3 #0.2 # dataset1: 0.2, dataset2 : -0.1
phi22 <- 0.6 #0.7

Phi <- matrix(c(phi11, phi12, phi21, phi22), 2, 2, byrow = T)

sigma_eta1 <- 1.5
sigma_eta2 <- 0.5
Sigma_eta <- diag(c(sigma_eta1, sigma_eta2))

sigma_eps1 <- 1 #0.01
sigma_eps2 <- 1 #0.02

Sigma_eps <- diag(c(sigma_eps1, sigma_eps2))

Tfin <- 1000
x1 <- c(0, 0)
X <- matrix(NA, nrow = length(x1), ncol = Tfin+1) # x_0:T
X[, 1] <- x1
Y <- matrix(NA, nrow = length(x1), ncol = Tfin) # y_1:T
# Y[, 1] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
set.seed(2022)
for (t in 1:Tfin) {
  X[, t+1] <- Phi %*% X[, t] + t(rmvnorm(1, c(0, 0), Sigma_eta))
  V <- diag(exp(X[, t+1]/2))
  Y[, t] <- V %*% t(rmvnorm(1, c(0, 0), Sigma_eps))
}

par(mfrow = c(2,1))
plot(X[1, ], type = "l")
plot(X[2, ], type = "l")

plot(Y[1, ], type = "l")
plot(Y[2, ], type = "l")

### Test spectral density matrix ###
univariate1 <- compute_whittle_likelihood_sv(y = Y[1, ],
                                            params = list(phi = Phi[1,1],
                                                          sigma_eta = sqrt(Sigma_eta[1,1])))
spec_dens1 <- univariate1$spec_dens_x
I_1 <- univariate1$periodogram

univariate2 <- compute_whittle_likelihood_sv(y = Y[2, ],
                                            params = list(phi = Phi[2,2],
                                                          sigma_eta = sqrt(Sigma_eta[2,2])))
spec_dens2 <- univariate2$spec_dens_x
I_2 <- univariate2$periodogram

## Fourier frequencies
k <- seq(-ceiling(Tfin/2)+1, floor(Tfin/2), 1)
k_in_likelihood <- k [k >= 1 & k <= floor((Tfin-1)/2)]
freq <- 2 * pi * k_in_likelihood / Tfin

# ## astsa package
Z <- log(Y^2) - rowMeans(log(Y^2))
fft_out <- mvspec(t(Z), detrend = F, plot = F)
I_multi <- fft_out$fxx

spec_dens_multi <- compute_whittle_likelihood_multi_sv(Y = Y, fourier_freqs = freq, periodogram = I_multi,
                                                       params = list(Phi = Phi,
                                                                     Sigma_eta = Sigma_eta))$spec_dens_X
spec_dens_multi_1 <- unlist(lapply(spec_dens_multi, function(x) x[1,1]))
spec_dens_multi_2 <- unlist(lapply(spec_dens_multi, function(x) x[2,2]))

## Compare spectral densities
par(mfrow = c(2,1))
plot(spec_dens1, type = "l")
lines(Re(spec_dens_multi_1), col = "red")

plot(spec_dens2, type = "l")
lines(Re(spec_dens_multi_2), col = "red")

## Check spectral density calculation using beyondWhittle package
spec_dens_beyondW <- psd_varma(freq = freq, ar = Phi, Sigma = Sigma_eta)
spec_dens_multi

## Compare periodograms: Check that I_multi[i,i] = I_uni[i]
I_multi[,,1]
I_1[1]
I_2[1]

## Test DFT ##
# ## astsa package
Z <- log(Y^2) - rowMeans(log(Y^2))
fft_out <- mvspec(t(Z), detrend = F, plot = F)
I_all <- fft_out$fxx

## Manual computation
k <- seq(-ceiling(Tfin/2) + 1, floor(Tfin/2), 1)
k_in_likelihood <- k[k >= 1 & k <= floor((Tfin-1)/2)]
freq <- 2 * pi * k_in_likelihood/Tfin

Z_list <- lapply(seq_len(ncol(Z)), function(i) Z[,i])
# J <- 0
I <- list()
J <- list()
for (j in 1:length(freq)) {
  mult_factor <- exp(-1i * freq[j] * (1:Tfin))
  J_elements <- mapply("*", Z_list, mult_factor, SIMPLIFY = F)
  J_mat <- matrix(unlist(J_elements), nrow = nrow(Z), ncol = ncol(Z))
  # J_t <- rowSums(J_mat)
  
  # J_t <- matrix(0, nrow(Z), ncol(Z))
  # for (t in 1:Tfin) {
  #   J_t[, t] <- Z[, t] * exp(-1i * freq[j] * t)
  # }
  # 
  # browser()
  J[[j]] <- rowSums(J_mat)
  I[[j]] <- 1/Tfin * J[[j]] %*% t(Conj(J[[j]]))
}

# fft_out <- fft(Z)
# test <- fft_out[, k_in_likelihood]
# test_list <- lapply(seq_len(ncol(test)), function(i) test[,i])
# I_fft <- lapply(test_list, function(x) 1/Tfin * x %*% t(Conj(x)))
