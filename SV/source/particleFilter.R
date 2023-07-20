particleFilter <- function(y, N = 500, iniState = 0, params) {
  
  obs <- log(y^2) - mean(log(y^2))
  Tfin <- length(obs)
  phi <- params$phi
  sigma_eta <- params$sigma_eta
  sigma_eps <- params$sigma_eps 
  
  # create state and weight matrices
  x_pf <- matrix(nrow = N, ncol = Tfin)
  weights <- norm_weights <- matrix(nrow =  N, ncol = Tfin)
  log_likelihood <- 0
  
  # intial (at t=1):
  # draw X from prior distribution
  
  x_pf[, 1] <- rnorm(N, 0, sqrt(sigma_eta^2 / (1 - phi^2)))
  # calculate weights, i.e. probability of evidence given sample from X
  E_log_eps2 <- digamma(1/2) + log(2)
  pseudo_xi <- obs[1] - x_pf[, 1] + E_log_eps2
  # weights[, 1] <- dchisq(exp(pseudo_xi), 1) * exp(pseudo_xi)
  weights[, 1] <- dgamma(exp(pseudo_xi), shape = 1/2, scale = 2*sigma_eps^2) * exp(pseudo_xi)
  
  # calculate likelihood
  log_likelihood <- log(mean(weights[, 1]))
  
  # normalise weights 
  norm_weights[, 1] <- weights[, 1]/sum(weights[, 1])
  
  # # normalise weights -- with better precision
  # log_weights <- log(weights[, 1])
  # maxWeight <- max(log_weights)
  # expWeights <- exp(log_weights - maxWeight) #exp(weights - maxWeights)
  # sumWeights <- sum(expWeights)
  # norm_weights[, 1] <- expWeights / sumWeights
  # norm_weights2[, 1] <- expWeights / sum(expWeights)

  # Compute likelihood -- with better precision
  # predictive_like <- maxWeight + log(sumWeights) - log(N)
  # log_likelihood <- log_likelihood + predictive_like
  
  # weighted resampling with replacement. This ensures that X will converge to the true distribution
  x_pf[, 1] <- sample(x_pf[, 1], replace = TRUE, size = N, prob = norm_weights[, 1]) 
  
  for (t in 2:Tfin) {
    # predict x_{t} from previous time step x_{t-1}
    # based on process (transition) model
    x_pf[, t] <- rnorm(N, phi * x_pf[, t-1], sigma_eta)
    # calculate  and normalise weights
    # weights[, t] <- dnorm(obs[t], x_pf[, t], sy)
    pseudo_xi <- obs[t] - x_pf[, t] + E_log_eps2
    # weights[, t] <- dchisq(exp(pseudo_xi), 1) * exp(pseudo_xi)
    weights[, t] <- dgamma(exp(pseudo_xi), shape = 1/2, scale = 2*sigma_eps^2) * exp(pseudo_xi)
    
    # estimate likelihood
    log_likelihood <- log_likelihood + log(mean(weights[, t]))
    
    # normalise weights
    norm_weights[, t] <- weights[, t]/sum(weights[, t])
    
    # weighted resampling with replacement
    x_pf[, t] <- sample(x_pf[, t], replace = TRUE, size = N, prob = norm_weights[, t]) 
  }
  
  x_means <- apply(x_pf, 2, mean)
  # x_quantiles <- apply(x_pf, 2, function(x) quantile(x, probs = c(0.025, 0.975)))
  # par(mfrow = c(1,1))
  # plot(x, type = "l")
  # lines(x_means, col = "red") # someone else's code
  # lines(x_estimate, col = "blue") # my new code
  # lines(pf_out$filteredState, col = "green") # my old code from last year
  
  return(list(state_mean = x_means, log_likelihood = log_likelihood))
  
}