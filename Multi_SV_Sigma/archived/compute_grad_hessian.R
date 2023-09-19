## Calculate the gradient and Hessian of the likelihood based on those samples
compute_grad_hessian <- tf_function(
  testf <- function(samples_tf, I_i, freq_i) {
    with (tf$GradientTape() %as% tape2, {
      with (tf$GradientTape(persistent = TRUE) %as% tape1, {
        
        A_samples_tf <- samples_tf[, 1:4]
        A_samples_tf <- tf$reshape(A_samples_tf, c(dim(A_samples_tf)[1], 2L, 2L))
        
        ## Construct Sigma_eta
        zero_vec <- tf$constant(rep(0, S), dtype = "float64")
        L_elements <- tf$concat(list(tf$reshape(tf$exp(samples_tf[, 5]), c(S, 1L)), 
                                     tf$reshape(zero_vec, c(S, 1L)),
                                     tf$reshape(samples_tf[, 7], c(S, 1L)),
                                     tf$reshape(tf$exp(samples_tf[, 6]), c(S, 1L))
        ), axis = 1L)
        L_tf <- tf$reshape(L_elements, c(S, 2L, 2L))
        Sigma_eta_samples_tf <- tf$linalg$matmul(L_tf, tf$transpose(L_tf, perm = c(0L, 2L, 1L)))
        
        ## Map A to Phi 
        Phi_samples_tf <- backward_map_tf(A_samples_tf, Sigma_eta_samples_tf)
        
        ## Reshape Phi and Sigma_eta
        # Phi_samples_tf <- tf$reshape(Phi_samples_tf, c(1L, dim(Phi_samples_tf)))
        # Sigma_eta_samples_tf <- tf$reshape(Sigma_eta_samples_tf, c(1L, dim(Sigma_eta_samples_tf)))
        
        # samples_tf <- tf$concat(list(Phi_samples_tf, Sigma_eta_samples_tf), 0L)
        # 
        # Phi_test <- samples_tf[1,,,]
        # Sigma_eta_test <- samples_tf[2,,,]
        
        
        # Spectral density matrix
        Phi_0_tf <- tf$eye(2L) #diag(2)
        # Phi_0_reshape <- tf$reshape(Phi_0_tf, c(1L, dim(Phi_0_tf)))
        # Phi_0_tiled <- tf$tile(Phi_0_reshape, c(S, 1L, 1L))
        Phi_1_tf <- Phi_samples_tf #[1,,]
        Theta_tf <- Phi_0_tf#tiled #diag(2)
        
        # freq_tf <- freq_i #freq[j] 
        # I_tf <- tf$Variable(I_i) #tf$constant(I_all[,,j])
        I_tf_reshaped <- tf$reshape(I_i, c(1L, dim(I_i)))
        I_tf_tiled <- tf$tile(I_tf_reshaped, c(S, 1L, 1L))
        
        Phi_mat <- tf$cast(Phi_0_tf, "complex128") - tf$multiply(tf$cast(Phi_1_tf, "complex128"), 
                                                                 tf$exp(tf$multiply(-1i, tf$cast(freq_i, "complex128"))))
        Phi_inv_tf <- tf$linalg$inv(Phi_mat)
        Phi_inv_H_tf <- tf$math$conj(tf$transpose(Phi_inv_tf, perm = c(0L, 2L, 1L))) # perm is to make sure transposes are done on the 2x2 matrix not on the batch dimension
        
        Theta_tf <- tf$cast(Theta_tf, "complex128")
        Sigma_eta_test <- tf$cast(Sigma_eta_samples_tf, "complex128")
        
        spec_dens_X_tf <- tf$linalg$matmul(tf$linalg$matmul(tf$linalg$matmul(tf$linalg$matmul(Phi_inv_tf, Theta_tf), Sigma_eta_test), Theta_tf), Phi_inv_H_tf)
        # tf$linalg$matmul(Theta_tf, 
        #                  tf$linalg$matmul(Sigma_eta_test, 
        #                                   tf$linalg$matmul(Theta_tf, Phi_inv_H_tf)))) 
        
        spec_dens_Xi_tf <- tf$multiply(pi^2/2, tf$eye(2L)) # tf$diag(pi^2/2, 2)
        spec_dens_Xi_tf <- tf$cast(spec_dens_Xi_tf, "complex128")
        
        spec_dens_tf <- spec_dens_X_tf + spec_dens_Xi_tf  
        
        part2_tf <- tf$linalg$trace(tf$linalg$matmul(tf$linalg$inv(spec_dens_tf), I_tf_tiled))
        
        det_spec_dens_tf <- tf$math$reduce_prod(tf$linalg$eigvals(spec_dens_tf), axis = 1L)
        part1_tf <- tf$math$log(det_spec_dens_tf)
        
        log_likelihood_tf <- -(part1_tf + part2_tf)
        log_likelihood_tf <- tf$math$real(log_likelihood_tf)
        
        # return(log_likelihood_tf)
      })
      grad_tf %<-% tape1$gradient(log_likelihood_tf, samples_tf)
      
      # grad %<-% tape1$gradient(log_likelihood, c(theta_tf[, 1], theta_tf[, 2]))
      # grad_tf <- tf$reshape(grad_tf, c(2L, dim(grad_tf[[1]])))
    })
    grad2_tf %<-% tape2$batch_jacobian(grad_tf, samples_tf)
    # grad2_tf %<-% tape2$jacobian(grad_tf, c(theta_phi, theta_eta))
    
    return(list(llh = log_likelihood_tf,
                grad = grad_tf,
                hessian = grad2_tf))
  },
  reduce_retracing=T
)