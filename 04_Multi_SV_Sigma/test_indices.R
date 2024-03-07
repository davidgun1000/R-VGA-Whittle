rm(list = ls())

library(tensorflow)
reticulate::use_condaenv("tf2.11", required = TRUE)
library(keras)
library(mvtnorm)

# List physical devices
gpus <- tf$config$experimental$list_physical_devices('GPU')

if (length(gpus) > 0) {
  tryCatch({
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    tf$config$experimental$set_virtual_device_configuration(
      gpus[[1]],
      list(tf$config$experimental$VirtualDeviceConfiguration(memory_limit=4096))
    )
    
    logical_gpus <- tf$config$experimental$list_logical_devices('GPU')
    
    print(paste0(length(gpus), " Physical GPUs,", length(logical_gpus), " Logical GPUs"))
  }, error = function(e) {
    # Virtual devices must be set before GPUs have been initialized
    print(e)
  })
}


# index_to_i_j_rowwise_nodiag <- function(k) {
#   kp <- k - 1
#   p  <- floor((sqrt(1 + 8 * kp) - 1) / 2)
#   i  <- p + 2
#   j  <- kp - p * (p + 1) / 2 + 1
#   c(i, j)
# }
# 
# index_to_i_j_rowwise_nodiag(5)
# 
# d <- 4
# numlist <- 1:(d*(d-1)/2)
# 
# mat <- diag(100, d)
# 
# for (num in 1:length(numlist)) {
#   ind <- index_to_i_j_rowwise_nodiag(num)
#   mat[ind[1], ind[2]] <- num
# }
# mat

### Tensorflow ###

# d <- 3
# param_dim <- d + d*(d-1)/2
# sample_mean <- rep(0, param_dim)
# sample_var <- diag(param_dim)
# S <- 10L
# samples <- rmvnorm(S, sample_mean, sample_var)
# samples_tf <- tf$Variable(samples)
# 
# # dim_rep <- tf$constant(rep(d, S), dtype = "float64")
# # dim_rep <- tf$reshape(dim_rep, c(dim(dim_rep), 1L))
# # test <- tf$concat(list(dim_rep, samples_tf), axis = 1L)
# 
# dim <- d
# vals <- samples_tf

fill_lower_tri_one_tf <- tf_function(
  fill_lower_tri_one <- function(dim, vals) {
    d <- as.integer(dim)
    # vals_tf <- tf$constant(vals, dtype = "float64")
    
    # sess = tf.InteractiveSession()
    # diag_mat <- tf$linalg$diag(tf$exp(vals_tf[(d+1):(2*d)]))
    diag_mat <- tf$linalg$diag(tf$exp(vals[1:d]))
    
    nlower <- d*(d-1)/2
    # x = tf$constant(1:nlower)
    numlower = vals[(d+1):(d+nlower)]
    ones = tf$ones(c(d, d), dtype="int64")
    mask_a = tf$linalg$band_part(ones, -1L, 0L)  # Upper triangular matrix of 0s and 1s
    mask_b = tf$linalg$band_part(ones, 0L, 0L)  # Diagonal matrix of 0s and 1s
    mask = tf$subtract(mask_a, mask_b) # Mask of upper triangle above diagonal
    
    zero = tf$constant(0L, dtype="int64")
    non_zero = tf$not_equal(mask, zero) #Conversion of mask to Boolean matrix
    indices = tf$where(non_zero) # Extracting the indices of upper trainagle elements
    
    out = tf$SparseTensor(indices, numlower, dense_shape = tf$cast(c(d,d), dtype="int64"))
    lower_tri = tf$sparse$to_dense(out)
    # dense = tf$print(dense, [dense], summarize=100)
    # sess.run(dense)
    L = diag_mat + lower_tri
    
    return(L)
  }
)


fill_lower_tri_tf <- tf_function(
  fill_lower_tri <- function(dim, vals) {
    d <- as.integer(dim)
    S <- as.integer(nrow(vals))
    # vals_tf <- tf$constant(vals, dtype = "float64")
    
    diag_mat <- tf$linalg$diag(tf$exp(vals[, 1:d]))
    diag_mat_tiled <- tf$tile(diag_mat, c(S, 1L, 1L))
      
    nlower <- as.integer(d*(d-1)/2)
    numlower = vals[, (d+1):(d+nlower)]
    numlower = tf$reshape(numlower, c(S*nlower, 1L))
    numlower = tf$squeeze(numlower) 
    # numlower = tf$reshape(numlower, -1L) # flatten into a vector
    # numlower = c(t(vals[, (d+1):(d+nlower)]))
    
    ones = tf$ones(c(d, d), dtype="int64")
    mask_a = tf$linalg$band_part(ones, -1L, 0L)  # Upper triangular matrix of 0s and 1s
    mask_b = tf$linalg$band_part(ones, 0L, 0L)  # Diagonal matrix of 0s and 1s
    mask = tf$subtract(mask_a, mask_b) # Mask of upper triangle above diagonal
    
    zero = tf$constant(0L, dtype="int64")
    non_zero = tf$not_equal(mask, zero) #Conversion of mask to Boolean matrix
    non_zero_tile <- tf$tile(non_zero, c(S, 1L))
    indices = tf$where(non_zero_tile) # Extracting the indices of upper triangular elements
    
    ## need to reshape indices here
    # S <- dim(vals)[1]
    # indices <- tf$reshape(indices, c(1L, dim(indices)))
    # batch_indices <-tf$tile(indices, c(S, 1L, 1L))
    shape <- tf$cast(c(S*d, d), dtype="int64")
    # shape_test <- tf$reshape(shape, c(dim(shape), 1L))
    # batch_shapes <- tf$tile(shape, c(S, 1L))
    out = tf$SparseTensor(indices, numlower, 
                          dense_shape = shape)
    lower_tri = tf$sparse$to_dense(out)
    lower_tri_reshaped = tf$reshape(lower_tri, c(S, d, d))
    # dense = tf$print(dense, [dense], summarize=100)
    # sess.run(dense)
    L = diag_mat + tf$cast(lower_tri_reshaped, dtype = "float64")
    
    return(L)
  }
)

d <- 3L
param_dim <- d + d*(d-1)/2

sample_mean <- rep(0, param_dim)
sample_var <- diag(param_dim)
S <- 10L
samples <- rmvnorm(S, sample_mean, sample_var)
samples_tf <- tf$Variable(samples)

Lmat <- fill_lower_tri_one_tf(d, samples_tf[1, ])
Lmat_all <- fill_lower_tri_tf(d, samples_tf)
