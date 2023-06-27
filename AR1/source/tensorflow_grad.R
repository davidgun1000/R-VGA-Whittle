## TensorFlow
rm(list = ls())

reticulate::use_condaenv("tf2.11", required = TRUE)
library(tensorflow)
library(keras)

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


## Generate AR(1) series
# set.seed(2023)
# x0 <- 1 #rnorm(1, 0, 1) 
# x <- c()
# x[1] <- x0
# 
# for (t in 2:n) {
#   x[t] <- phi * x[t-1] + rnorm(1, 0, sigma_e) 
# }

################################################################################
# x <- tf$Variable(3)
# 
# with(tf$GradientTape() %as% tape, {
#   y <- x ^ 2
# })
# 
# dy_dx <- tape$gradient(y, x)
# dy_dx

# ## Harder case
# w <- tf$Variable(tf$random$normal(c(3L, 2L)), name = 'w')
# b <- tf$Variable(tf$zeros(2L, dtype = tf$float32), name = 'b')
# x <- as_tensor(1:3, "float32", shape = c(1, 3))
# 
# with(tf$GradientTape(persistent = TRUE) %as% tape, {
#   y <- tf$matmul(x, w) + b
#   loss <- mean(y ^ 2)
# })
# 
# c(dl_dw, dl_db) %<-% tape$gradient(loss, c(w, b))
# 
# u <- tf$Variable(3)
# v <- tf$Variable(4)
# 
# with(tf$GradientTape() %as% tape, {
#   y <- tf$math$multiply(tf$math$tanh(u), tf$math$cos(v))
# })
# 
# c(dy_du, dy_dv) %<-% tape$gradient(y, c(u, v))

## Jacobian

# x = tf$linspace(-10.0, 10.0, 200L+1L)
# delta = tf$Variable(0.0)
# 
# with (tf$GradientTape() %as% tape, {
#   y = tf$nn$sigmoid(x+delta)
# })
# 
# dy_dx = tape$jacobian(y, delta)


## Test the spectral density stuff
n <- 1000
omega <- 2 * pi / n
theta_phi <- tf$Variable(atanh(0.9))
theta_sigma <- tf$Variable(log(0.5^2))
freq <- tf$Variable(omega)

## Fourier frequencies
# k_in_likelihood <- seq(1, floor((n-1)/2)) 
# freq <- 2 * pi * k_in_likelihood / n

## Fourier transform of the series
# fourier_transf <- fft(x)
# periodogram <- 1/n * Mod(fourier_transf)^2
# I <- periodogram[k_in_likelihood + 1]
# I_tf <- tf$Variable(I[[1]])

# for (i in 1:10) {
  I_tf <- tf$Variable(1)
  
  S <- 3L
  sigma_e <- runif(S, 0.5, 0.9)
  theta_sigma <- log(sigma_e^2)
  phi <- runif(S, 0.5, 0.9)
  theta_phi <- atanh(phi)
  
  # theta_sigma <- c(1, 2)
  # theta_phi <- c(2, 3)
  
  # should combine pairs of phi and sigma here?
  theta_sigma_tf <- tf$Variable(theta_sigma)
  theta_phi_tf <- tf$Variable(theta_phi)
  
  # theta_tf <- tf$Variable(cbind(theta_sigma, theta_phi))
  # and the write the likelihood function in terms of matrices
  # how to write the spectral density in that form?
  compute_grad <- tf_function(
  testf <- function(theta_phi_tf, theta_sigma_tf, I_tf, freq) {
      with (tf$GradientTape() %as% tape2, {
        with (tf$GradientTape(persistent = TRUE) %as% tape1, {
          spec_dens <- tf$math$divide(tf$math$exp(theta_sigma_tf), 1 + tf$math$square(tf$math$tanh(theta_phi_tf)) -
                                        tf$math$multiply(2, tf$math$multiply(tf$math$tanh(theta_phi_tf), tf$math$cos(freq))))
          
          log_likelihood <- - tf$math$log(spec_dens) - tf$multiply(I_tf, tf$math$reciprocal(spec_dens))
          # tape1$watch(c(theta_phi_tf, theta_sigma_tf))
          # log_likelihood <- tf$math$pow(theta_phi_tf, 3) + tf$math$pow(theta_sigma_tf, 3)
          # log_likelihood <- tf$math$pow(theta_tf[, 1], 3) + tf$math$pow(theta_tf[, 2], 3)
          
        })  
        # c(grad_theta_phi, grad_theta_sigma) %<-% tape1$gradient(log_likelihood, c(theta_phi_tf, theta_sigma_tf))
        grad %<-% tape1$gradient(log_likelihood, c(theta_phi_tf, theta_sigma_tf))
        
        grad <- tf$reshape(grad, c(2L, dim(grad[[1]])))
      })
      grad2 %<-% tape2$jacobian(grad, c(theta_phi_tf, theta_sigma_tf))
      
  #     browser()
  #     
      return(list(grad = grad,
                  hessian = grad2))
    }
  )
  
  out <- compute_grad(theta_phi_tf, theta_sigma_tf, I_tf, freq)
  
  
  # grad2_theta_phi %<-% tape2$gradient(grad, c(theta_phi_tf, theta_sigma_tf))
  # grad2_theta_phi %<-% tape2$gradient(grad, theta_phi_tf)
  
# }

# grads <- out$grad
# hessians <- out$hessian
# ## need to then reshape these into the right grads and hessians
# 
# ## gradients
# grad_first <- grads[, 1] #
# grad_second <- grads[, 2]
# 
# E_grad <- rowMeans(as.matrix(grads, 2, 2))
# 
# ## batch-extract diagonals, and then extract first element of diagonal as grad2_phi(1),
# ## second element as grad2_phi(2) etc
# grad2_phi <- diag(as.matrix(hessians[[1]][1,,], S, S)) #grad2_phi
# grad2_phi_sigma <- diag(as.matrix(hessians[[1]][2,,], S, S)) #grad2_phi_sigma
# grad2_sigma_phi <- diag(as.matrix(hessians[[2]][1,,], S, S)) #grad2_sigma_phi
# grad2_sigma <- diag(as.matrix(hessians[[2]][2,,], S, S)) #grad2_sigma
# 
# # take mean of each element in Hessian, then put them together in a 2x2 matrix E_hessian
# E_grad2_phi <- mean(grad2_phi)
# E_grad2_phi_sigma <- mean(grad2_phi_sigma)
# E_grad2_sigma_phi <- mean(grad2_sigma_phi)
# E_grad2_sigma <- mean(grad2_sigma)
# 
# E_hessian <- matrix(c(E_grad2_phi, E_grad2_phi_sigma, E_grad2_sigma_phi, E_grad2_sigma), 2, 2, byrow = T)


# ######## Test deriv() #######
# # Expression or formula
# f = expression(x^2 + 5*x + 1)
# 
# # Derivative
# cat("Using deriv() function:\n")
# print(deriv(f, "x"))
# 
# cat("\nUsing D() function:\n")
# print(D(f, 'x'))
# 
# spec_dens <- expression(exp(theta_sigma) / (1 + tanh(theta_phi)^2 - 2 * tanh(theta_phi) * cos(freq)))
# grad_spec_dens <- deriv(spec_dens, "theta_sigma")
# 
# jac <- function(exprs, vars) {
#   ## capture the elements in a non-evaluated form
#   ee <- substitute(exprs)
#   ## assumes that the expressions are combined via list(), c(), etc.
#   ## so we remove this part of the expression with [-1] and apply derivs() to each remaining element
#   exprs <- lapply(ee[-1], deriv, vars, hessian = TRUE)
#   function(...) {
#     ## apply eval(), with the named values found in ..., to the individual expressions
#     results <- lapply(exprs, eval, list(...))
#     ## extract the gradient value from each result and combine into a matrix
#     do.call(rbind, lapply(results, attr, "gradient"))
#   }
# }
# f_jac <- jac(c(sin(x), cos(x), atan(y/x), tan(x+y)), c("x", "y"))
# f_jac(x = 3, y = 2)
# 
# g <- theta_phi^3 + theta_sigma^3
# g_jac <- jac(g, c("theta_phi", "theta_sigma"))
# g_jac(theta_phi = 2, theta_sigma = 1)
# 
# grad_theta_phi <- deriv(substitute(tanh(theta_phi)^2 + tanh(theta_sigma)), 
#                         c("theta_phi", "theta_sigma"), hessian = TRUE)
# # grad2_theta_phi <- deriv(grad_theta_phi, c("theta_phi", "theta_sigma"))
# theta_phi <- 2
# theta_sigma <- 1
# test <- attributes(eval(grad_theta_phi))
# as.matrix(test$hessian, 2L, 2L)


## Try Deriv package
# library(Deriv)
# freq <- 2*pi/1000
# I_i <- 1
# g <- function(theta_phi, theta_sigma, omega_k, I_k) { 
#   spec_dens <- exp(theta_sigma)/( 1 + tanh(theta_phi)^2 - 2 * tanh(theta_phi) * cos(freq))
#   
#   log_likelihood <- - log(spec_dens) - I_k / spec_dens
# }
# 
# testD <- Deriv(g, c("theta_phi", "theta_sigma"))
# testD(2,1, I_k = I_i, omega = freq)
# testgrad <- mapply(testD, theta_phi = c(2,3), theta_sigma = c(1,2), omega = freq, I_k = I_i) 
# 
# testD2 <- Deriv(testD, c("theta_phi", "theta_sigma"))
# testgrad2 <- mapply(testD2, theta_phi = c(2,3), theta_sigma = c(1,2), omega = freq, I_k = I_i) 
