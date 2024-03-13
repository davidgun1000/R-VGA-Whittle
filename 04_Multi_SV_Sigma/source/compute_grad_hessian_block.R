## Calculate the gradient and Hessian of the likelihood based on those samples
compute_grad_hessian <- tf_function(
  compute_grad_hessian <- function(samples_tf, I_i, freq_i, blocksize, use_cholesky = F,
                                   transform = "logit") {
    with (tf$GradientTape() %as% tape2, {
      with (tf$GradientTape(persistent = TRUE) %as% tape1, {
        d <- as.integer(dim(I_i)[1])
        S <- as.integer(dim(samples_tf)[1])
        # blocksize <- as.integer(length(freq_i))
      
        if (use_cholesky) {
          param_dim <- d + (d*(d-1)/2 + d) # d AR parameters, 
          # d*(d-1)/2 + d parameters from the lower Cholesky factor of Sigma_eta
        } else {
          param_dim <- d^2 + d
        }
        
        if (transform == "arctanh") {
          Phi_samples_tf <- tf$linalg$diag(tf$math$tanh(samples_tf[, 1:d]))
        } else {
          Phi_samples_tf <- tf$linalg$diag(tf$math$reciprocal(1 + tf$math$exp(-samples_tf[, 1:d])))
        }
        
        ## Construct Sigma_eta
        if (use_cholesky) {
          Lsamples_tf <- fill_lower_tri(d, samples_tf[, (d+1):param_dim])
          Sigma_eta_samples_tf <- tf$linalg$matmul(Lsamples_tf, tf$transpose(Lsamples_tf, perm = c(0L, 2L, 1L)))
          
        } else {
          Sigma_eta_samples_tf <- tf$linalg$diag(tf$exp(samples_tf[, (d^2+1):param_dim]))
        }
        
        # Spectral density matrix
        Phi_0_tf <- tf$cast(tf$eye(d), "complex128") #diag(2)
        Phi_1_tf <- tf$cast(Phi_samples_tf, "complex128") #[1,,]
        Theta_tf <- Phi_0_tf#tiled #diag(2)
       
        if (length(dim(I_i)) == 2) {
          I_tf_reshaped <- tf$reshape(I_i, c(1L, 1L, dim(I_i)))
        } else {
          I_i <- tf$transpose(I_i, perm = c(2L, 0L, 1L))
          I_tf_reshaped <- tf$reshape(I_i, c(1L, dim(I_i)))
        }
        I_tf_tiled <- tf$tile(I_tf_reshaped, c(S, 1L, 1L, 1L))
        
        freq_i <- tf$reshape(freq_i, c(1L, blocksize, 1L, 1L))
        freq_tiled <- tf$tile(freq_i, c(S, 1L, 1L, 1L))
        
        Phi_0_tf <- tf$reshape(Phi_0_tf, c(1L, 1L, dim(Phi_0_tf)))
        Phi_0_tiled <- tf$tile(Phi_0_tf, c(S, blocksize, 1L, 1L))
        
        
        Phi_1_tf <- tf$reshape(Phi_1_tf, c(S, 1L, d, d))
        Phi_1_tiled <- tf$tile(Phi_1_tf, c(1L, blocksize, 1L, 1L))
        
        Phi_mat <- tf$cast(Phi_0_tiled, "complex128") - 
                    tf$multiply(tf$cast(Phi_1_tiled, "complex128"), 
                                 tf$exp(tf$multiply(-1i, tf$cast(freq_tiled, "complex128"))))
        
        Phi_inv_tf <- tf$linalg$inv(Phi_mat)
        Phi_inv_H_tf <- tf$math$conj(tf$linalg$matrix_transpose(Phi_inv_tf)) 
        
        
        
        Theta_tf <- tf$cast(Theta_tf, "complex128")
        Sigma_eta_samples_tf <- tf$cast(Sigma_eta_samples_tf, "complex128")
        Sigma_eta_samples_tf <- tf$reshape(Sigma_eta_samples_tf, c(S, 1L, d, d))
        Sigma_eta_tiled <- tf$tile(Sigma_eta_samples_tf, c(1L, blocksize, 1L, 1L))
        # spec_dens_X_tf <- tf$linalg$matmul(tf$linalg$matmul(tf$linalg$matmul(tf$linalg$matmul(Phi_inv_tf, Theta_tf), 
        #                                                                      Sigma_eta_test), 
        #                                                     Theta_tf), 
        #                                    Phi_inv_H_tf)
       
        spec_dens_X_tf <- tf$linalg$matmul(tf$linalg$matmul(tf$linalg$matmul(tf$linalg$matmul(Phi_inv_tf, Phi_0_tiled), #Theta_tf), 
                                                                             Sigma_eta_tiled), 
                                                            Phi_0_tiled), 
                                           Phi_inv_H_tf)
        
        spec_dens_Xi_tf <- tf$multiply(pi^2/2, tf$eye(d)) # tf$diag(pi^2/2, 2)
        spec_dens_Xi_tf <- tf$reshape(tf$cast(spec_dens_Xi_tf, "complex128"), c(1L, 1L, dim(spec_dens_Xi_tf)))
        spec_dens_Xi_tf <- tf$tile(spec_dens_Xi_tf, c(S, blocksize, 1L, 1L))
        
        spec_dens_tf <- spec_dens_X_tf + spec_dens_Xi_tf  
        
        part2_tf <- tf$linalg$trace(tf$linalg$matmul(tf$linalg$inv(spec_dens_tf), I_tf_tiled))
        
        det_spec_dens_tf <- tf$math$reduce_prod(tf$linalg$eigvals(spec_dens_tf), axis = 2L)
        part1_tf <- tf$math$log(det_spec_dens_tf)
        
        log_likelihood_tf <- -(part1_tf + part2_tf)
        log_likelihood_tf <- tf$math$reduce_sum(log_likelihood_tf, 1L)
        log_likelihood_tf <- tf$math$real(log_likelihood_tf)
        
        # return(log_likelihood_tf)
      })
      grad_tf %<-% tape1$gradient(log_likelihood_tf, samples_tf)
      
    })
    grad2_tf %<-% tape2$batch_jacobian(grad_tf, samples_tf)
    
    return(list(Phi_samples_tf = Phi_samples_tf,
                Sigma_eta_samples_tf = Sigma_eta_samples_tf,
                llh = log_likelihood_tf,
                grad = grad_tf,
                hessian = grad2_tf))
  } ,
  reduce_retracing=T
)

# fill_lower_tri_tf <- tf_function(
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
# )