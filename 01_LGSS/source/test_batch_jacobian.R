setwd("~/R-VGA-Whittle/LGSS/")
rm(list = ls())

library(mvtnorm)
# library(coda)
# library(Deriv)
library(tensorflow)
reticulate::use_condaenv("tf2.11", required = TRUE)
library(keras)

# source("./source/run_rvgaw_lgss_tf.R")
# source("./source/run_mcmc_lgss_allparams.R")
# source("./source/compute_kf_likelihood.R")
# source("./source/compute_whittle_likelihood_lgss.R")
# source("./source/compute_whittle_likelihood_lb.R")
# source("./source/update_sigma.R")

################## Some code to limit tensorflow memory usage ##################

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

## Example batch jacobian
S <- 2L
x = tf$random$normal(c(S, 3L))

layer1 = tf$keras$layers$Dense(S, activation=tf$nn$elu)
layer2 = tf$keras$layers$Dense(3, activation=tf$nn$elu)

with (tf$GradientTape(persistent=TRUE, watch_accessed_variables=FALSE) %as% tape, {
    tape$watch(x)
    y = layer1(x)
    y = layer2(y)
  }
  # y$shape
)

j = tape$jacobian(y, x)
j$shape

## more efficient batch jacobian computation when the grads are independent:
jb = tape$batch_jacobian(y, x)
jb$shape


############### More advanced example

sigma_eps <- 0.5 # measurement error var
sigma_eta <- 0.7 # process error var
phi <- 0.9

n <- 1000
print("Generating data...")
x <- c()
x[1] <- 1
set.seed(2023)
for (t in 2:n) {
  x[t] <- phi * x[t-1] + rnorm(1, 0, sigma_eta)
}

# Generate observations y_1:T
y <- x + rnorm(n, 0, sigma_eps)

## Fourier frequencies
k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
freq <- 2 * pi * k_in_likelihood / n

## Fourier transform of the observations
fourier_transf <- fft(y)
periodogram <- 1/n * Mod(fourier_transf)^2
I <- periodogram[k_in_likelihood + 1]

freq_i_tf <- tf$constant(freq[1])
I_i_tf <- tf$constant(I[[1]]) 

## Computing gradient and Hessian

mu <- rep(0, 3)
P <- diag(3)
S <- 2L
samples <- rmvnorm(S, mu, P)
samples_tf <- tf$Variable(samples, dtype = "float32")
# theta_phi_s <- tf$Variable(samples[, 1])
# theta_eta_s <- tf$Variable(samples[, 2])
# theta_eps_s <- tf$Variable(samples[, 3])

compute_grad_arctanh_test <- tf_function(
  testf <- function(samples_tf, I_i, freq_i) {
    log_likelihood_tf <- 0
    with (tf$GradientTape() %as% tape2, {
      with (tf$GradientTape(persistent = TRUE) %as% tape1, {
        
        phi_s <- tf$math$tanh(samples_tf[, 1])
        spec_dens_x_tf <- tf$math$divide(tf$math$exp(samples_tf[, 2]), 1 + tf$math$square(phi_s) -
                                           tf$math$multiply(2, tf$math$multiply(phi_s, tf$math$cos(freq_i))))

        ## add spec_dens_eps here
        spec_dens_eps_tf <- tf$math$exp(samples_tf[, 3])

        ## then
        spec_dens_y_tf <- spec_dens_x_tf + spec_dens_eps_tf

        log_likelihood_tf <- - tf$math$log(spec_dens_y_tf) - tf$multiply(I_i, tf$math$reciprocal(spec_dens_y_tf))
        
      })
      # vars <- tf$reshape(cbind(theta_phi_s, theta_eta_s, theta_eps_s), c(length(theta_phi_s), 3L))
      grad_tf %<-% tape1$gradient(log_likelihood_tf, samples_tf)
      # grad_tf %<-% tape1$gradient(log_likelihood_tf, c(theta_phi_s, theta_eta_s, theta_eps_s))
      
      # grad_tf <- tf$reshape(tf$transpose(grad_tf), c(dim(grad_tf[[1]]), 3L))
    })
    # grad2_tf %<-% tape2$jacobian(grad_tf, c(theta_phi_s, theta_eta_s, theta_eps_s))
    
    # vars <- tf$reshape(c(theta_phi_s, theta_eta_s, theta_eps_s), dim(grad_tf))
    grad2_tf %<-% tape2$batch_jacobian(grad_tf, samples_tf)
    # grad2_tf %<-% tape2$batch_jacobian(grad_tf, vars)
    


  return(list(log_likelihood = log_likelihood_tf,
              grad = grad_tf,
              hessian = grad2_tf))
  }
)

tf_out_test <- compute_grad_arctanh_test(samples_tf, I_i_tf, freq_i_tf)
