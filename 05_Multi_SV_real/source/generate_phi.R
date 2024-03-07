rm(list = ls())
setwd("~/R-VGA-Whittle/Multi_SV/")

source("./source/map_functions.R")

d <- 3
A <- matrix(rnorm(d^2), d, d)

if (d == 2) {
  Sigma_eta <- diag(c(0.9, 1.5))
} else {
  Sigma_eta <- diag(c(0.9, 1.5, 1.2))
}

Phi <- backward_map(A, Sigma_eta)

Phi_rounded <- round(Phi, digits = 1)
print(Phi_rounded)