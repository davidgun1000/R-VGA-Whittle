rm(list = ls())
setwd("~/R-VGA-Whittle/Multi_SV_Sigma/")

library(cmdstanr)
stan_file <- "./source/test_array.stan"

original_array <- array(1:24 *(1i), dim = c(2, 3, 4))
re_matrices <- lapply(1:4, function(i) Re(original_array[,,i]))
im_matrices <- lapply(1:4, function(i) Im(original_array[,,i]))
# periodogram_array <- array(NA, dim = c(4, 2, 3))
# for(q in 1:4) {
#   periodogram_array[q,,] <- original_array[,,q]
# }

list_of_matrices <- lapply(1:4, function(x) original_array[,,x])

array_data <- list(N = 2, J = 3, Q = 4,
                  #  data_matrix = list_of_matrices,
                   re_matrices = re_matrices,
                   im_matrices = im_matrices)
                  # data_matrix = array(1:16, dim = c(4, 2, 2))); #original_array); #,
                  # data_matrix2 = array(1:16, dim = c(4, 2, 2)),
                  # data_vector = array(1:8, dim = c(4, 2)));

array_model <- cmdstan_model(
  stan_file,
  cpp_options = list(stan_threads = TRUE)
)



fit_stan_array <- array_model$sample(
  array_data,
  chains = 1,
  threads = parallel::detectCores(),
  refresh = 1,
  iter_warmup = 1,
  iter_sampling = 1
)
