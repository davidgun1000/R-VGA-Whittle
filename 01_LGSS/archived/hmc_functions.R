#' @title Hamiltonian MC sampler
#' @param U potential energy function.
#' @param dUdq derivative of potential energy function with respect to position.
#' @param M the mass matrix (i.e.,~the covariance matrix in the kinetic energy function)
#' @param eps_gen a function which generates the leapfrog step-size \eqn{\epsilon}
#' @param L number of leapfrog steps per proposal
#' @param lower vector of lower constraints on the variables
#' @param upper vector of upper constraints on the variables
#' @examples
#'
#' #' ### Model setup -- 2d correlated Gaussian
#' S <- matrix(c(1,-0.98,-0.98,1),2,2)
#' n <- nrow(S)
#' Q <- chol2inv(chol(S))
#' cholQ <- chol(Q)
#' U <- function(q) 0.5 * crossprod(cholQ %*% q)
#' dUdq <- function(q) Q %*% q
#' M <- diag(1,n)
#' 
#' ### Sampler parameters -- set eps and L according to eigenvalues of covariance matrix
#' E <- eigen(S)
#' eps_gen <- function() round(min(E$values),2)
#' L = as.integer(max(E$values)/eps_gen())
#' print(paste0("eps = ",eps_gen(),". L = ",L))
#' sampler <- hmc_sampler(U = U,dUdq = dUdq, M = M,eps_gen = eps_gen,L = L)
#' 
#' ### Now sample
#' N <- 1000
#' q <- matrix(0,n,N)
#' for(i in 2:N) q[,i] <- sampler(q = q[,(i-1)])
#' plot(t(q),ylim = c(-4,4),xlim=c(-4,4))
#' #' 

hmc_sampler <- function(M,eps_gen,L,lower=NULL,upper=NULL, data, I, freq, prior_mean, prior_var) {
# hmc_sampler <- function(U,dUdq,M,eps_gen,L,lower=NULL,upper=NULL) {
  .check_args(#U=U, dUdq = dUdq, 
              M=M, eps_gen = eps_gen, L=L,lower=lower,upper=upper)
  cholM <- chol(M)
  Minv <- chol2inv(cholM)
  function(q) {
    stopifnot(length(q) == nrow(M))
    .hmc_sample (q, #U=U, dUdq = dUdq, 
                  Minv = Minv, cholM = cholM, eps_gen=eps_gen, L=L,lower=lower,upper=upper,
                  data, I, freq, prior_mean, prior_var)
  }
}


#' @title Leapfrog steps
#' @param qp data frame with columns \code{q} and \code{p} denoting the position and momentum of the particles respectively
#' @param dUdq derivative of potential energy
#' @param Minv inverse of the mass matrix
#' @param eps step-size
#' @param L number of steps to simulate
#' @examples
#' nt <- 30
#' qp <- data.frame(q=0,p=1)
#' eps_gen <- function() 0.3
#' Minv = matrix(1,1,1)
#' 
#' dUdq <- function(q) q
#' 
#' for(i in 1:(nt-1)) 
#'   qp[i+1,] <- leapfrog(qp = qp[1,], dUdq = dUdq, Minv = Minv,eps = eps_gen(),L=i)
#' 
#' plot(qp$q,qp$p)
leapfrog <- function(qp,
                      # dUdq,
                      Minv,eps,L=1L,lower=NULL,upper=NULL, I, freq) {
  
  q <- qp$q
  p <- qp$p
  dUdq_q <- compute_dUdq(q, I, freq)
  p <- p - eps/2 * dUdq_q

  # p <- p - eps/2 * dUdq(q)
  for(i in 1:L) {
    q <- as.vector(q + eps*Minv %*% p)
    if(!(is.null(lower))) {
      q_id <- which(q < lower)
      q[q_id] <- lower[q_id] + (lower[q_id] - q[q_id])
      p[q_id] <- -p[q_id]
    }
    
    if(!(is.null(upper))) {
      q_id <- which(q > upper)
      q[q_id] <- upper[q_id] - (q[q_id] - upper[q_id])
      p[q_id] <- -p[q_id]
    }
    
    dUdq_q <- compute_dUdq(q, I, freq)
    
    if(!(i==L)) p <- p - eps*dUdq_q #dUdq(q) 
  }
  p <- p - eps * dUdq_q/2
  
  data.frame(q=q,p=p)
}


.is.diag <- function(X) {
  all(X[lower.tri(X)] == 0, X[upper.tri(X)] == 0)
}
.sample_momentum <- function(cholM) {
  t(cholM) %*% rnorm(n = nrow(cholM))
}

.propose_state <- function(qp,
                            # U,dUdq,
                            Minv,eps,L,lower,upper,I,freq) {
  
  #Run the dynamics
  qp <- leapfrog(qp = qp, 
                  # dUdq = dUdq,
                  Minv = Minv,eps = eps,L=L,lower=lower,upper=upper, I, freq)
  # Negate momentum for symmetric proposal
  qp$p <- -qp$p
  qp
}

.hmc_sample <- function(q,
                        # U,dUdq,
                        Minv,cholM,eps_gen,L,lower,upper, 
                        data, I, freq, prior_mean, prior_var) {
  
  current_q <- q
  current_p <- .sample_momentum(cholM = cholM)
  current_U <- compute_U(current_q, data, I, freq, prior_mean, prior_var)# U(current_q)
  current_K <- 0.5 * t(current_p) %*% Minv %*% current_p 
  qp <- data.frame(q=current_q, p = current_p)
  


  proposed_qp <- .propose_state(qp,
                                # U = U,dUdq = dUdq,
                                Minv = Minv,
                                eps = eps_gen(),L = L,lower=lower,upper=upper,
                                I = I, freq = freq)
  
  proposed_U <- compute_U(proposed_qp$q, data, I, freq, prior_mean, prior_var) #U(proposed_qp$q)
  proposed_K <- 0.5 * t(proposed_qp$p) %*% Minv %*% proposed_qp$p
  
  if (is.na(current_U - proposed_U + current_K - proposed_K)) {
    browser()
  }

  if(log(runif(1)) < current_U - proposed_U + current_K - proposed_K) {
    return(proposed_qp$q)
  } else {
    return(current_q)
  }
}

.check_args <- function(qp=data.frame(q=c(0,0),p=c(1,0)),
                        # U = function(){},
                        # dUdq = function(){},
                        M = matrix(c(1,0,0,1),2,2),
                        eps_gen=function() 0.1,
                        L=10L,
                        lower=NULL,
                        upper=NULL) {
  
  stopifnot(is.data.frame(qp))
  stopifnot(names(qp) == c("q","p"))
  # stopifnot(is.function(U))
  # stopifnot(is.function(dUdq))
  stopifnot(is.matrix(M))
  stopifnot(is.function(eps_gen))
  stopifnot(is.integer(L))
  stopifnot(L > 0)
  stopifnot(is.null(lower) | is.numeric(lower))
  stopifnot(is.null(upper) | is.numeric(upper))
  if(!(is.null(lower)) & !.is.diag(M)) stop("Constraints not implemented for non-diagonal M")
  if(!(is.null(upper)) & !.is.diag(M)) stop("Constraints not implemented for non-diagonal M")
  
}

################################################################################

## Functions for computing U and dUdq
compute_U <- function(q, y, I, freq, prior_mean, prior_var) {
  log_like <- compute_whittle_likelihood_lgss(params = q, I = I, freq = freq)
  # log_like2 <- compute_log_likelihood_arctanh(theta_phi = q[1], theta_eta = q[2], theta_eps = q[3], I = I, freq = freq)
  
  log_prior <- compute_prior(q, prior_mean, prior_var)
  
  return(log_like + log_prior)
} 

compute_dUdq <- function(q, I, freq) {
  tf_out <- compute_grad_arctanh(tf$Variable(q), tf$Variable(I), tf$Variable(freq))
  grad_log_like <- as.vector(tf_out$grad)
  # test <- Deriv(compute_log_likelihood_arctanh, c("theta_phi", "theta_eta", "theta_eps"))
  
  grad_log_prior <- compute_grad_prior(q, prior_mean, prior_var)

  return(grad_log_like + grad_log_prior)
}  

compute_log_likelihood_arctanh <- function(theta_phi, theta_eta, theta_eps, I, freq) { # same as the one below but parameters are listed as separate args
  
  phi <- tanh(theta_phi)
  sigma_eta <- sqrt(exp(theta_eta))
  sigma_eps <- sqrt(exp(theta_eps))
  
  spec_dens_x <- sigma_eta^2/( 1 + phi^2 - 2 * phi * cos(freq))
  spec_dens_eps <- sigma_eps^2 #/ (2*pi)
  spec_dens_y <- spec_dens_x + spec_dens_eps
  llh <- - log(spec_dens_y) - I / spec_dens_y
  llh <- sum(llh)
  # return(llh)
}

compute_whittle_likelihood_lgss <- function(params, I, freq) {
  n <- length(y)
  phi <- tanh(params[1]) #params$phi
  sigma_eta <- sqrt(exp(params[2])) #params$sigma_eps
  sigma_eps <- sqrt(exp(params[3])) #params$sigma_eta

  spectral_dens_x <- sigma_eta^2 / (1 + phi^2 - 2 * phi * cos(freq))
  spectral_dens_eps <- sigma_eps^2 #/ (2*pi)
  spectral_dens_y <- spectral_dens_x + spectral_dens_eps
  
  ## Calculate the Whittle likelihood
  log_whittle <- - log(spectral_dens_y) - I / spectral_dens_y
  log_whittle <- sum(log_whittle)
  
  return(log_whittle)
}


# compute_grad_arctanh <- tf_function(
  compute_grad_arctanh <- function(params_tf, I, freq) {
    log_likelihood_tf <- 0
    # with (tf$GradientTape() %as% tape2, {
    with (tf$GradientTape(persistent = TRUE) %as% tape1, {
      
      # phi_s <- params_tf[1] #
      # sigma_eta2_s <- params_tf[2]^2
      phi_s <- tf$math$tanh(params_tf[1])
      sigma_eta2_s <- tf$math$exp(params_tf[2])
      spec_dens_x_tf <- tf$math$divide(sigma_eta2_s, 1 + tf$math$square(phi_s) -
                                         tf$math$multiply(2, tf$math$multiply(phi_s, tf$math$cos(freq))))
      
      ## add spec_dens_eps here
      # sigma_eps2_s <- params_tf[3]^2 
      sigma_eps2_s <- tf$math$exp(params_tf[3])
      spec_dens_eps_tf <- sigma_eps2_s
      
      ## then
      spec_dens_y_tf <- spec_dens_x_tf + spec_dens_eps_tf
      
      log_likelihood_tf <- - tf$math$log(spec_dens_y_tf) - tf$multiply(I, tf$math$reciprocal(spec_dens_y_tf))
      log_likelihood_tf <- tf$reduce_sum(log_likelihood_tf)
    })
    # vars <- tf$reshape(cbind(theta_phi_s, theta_eta_s, theta_eps_s), c(length(theta_phi_s), 3L))
    grad_tf %<-% tape1$gradient(log_likelihood_tf, params_tf)
    # grad_tf %<-% tape1$gradient(log_likelihood_tf, c(theta_phi_s, theta_eta_s, theta_eps_s))
    
    # grad_tf <- tf$reshape(tf$transpose(grad_tf), c(dim(grad_tf[[1]]), 3L))
    # })
    # # grad2_tf %<-% tape2$jacobian(grad_tf, c(theta_phi_s, theta_eta_s, theta_eps_s))
    # 
    # # vars <- tf$reshape(c(theta_phi_s, theta_eta_s, theta_eps_s), dim(grad_tf))
    # grad2_tf %<-% tape2$batch_jacobian(grad_tf, samples_tf)
    # # grad2_tf %<-% tape2$batch_jacobian(grad_tf, vars)
    # 
    # E_grad_tf <- tf$reduce_mean(grad_tf, 0L)
    # E_hessian_tf <- tf$reduce_mean(grad2_tf, 0L)
    
    return(list(log_likelihood = log_likelihood_tf,
                grad = grad_tf))
                # hessian = grad2_tf,
                # E_grad = E_grad_tf))
                # E_hessian = E_hessian_tf))
  }
# )

compute_prior <- function(params, prior_mean, prior_var) {
  log_prior <- dmvnorm(params, prior_mean, prior_var, log = TRUE)
  return(log_prior)
}

compute_grad_prior <- function(params, prior_mean, prior_var) {
  grad_log_prior <- solve(prior_var, prior_mean - params)
  return(grad_log_prior)
}