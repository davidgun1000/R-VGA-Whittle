particleFilter <- function(y, N, iniState, param) {
  
  nT <- length(y) - 1
  particles <- matrix(NA, nrow = N, ncol = nT + 1)
  weights <- normalisedW <- matrix(NA, nrow = N, ncol = nT + 1)
  indices <- matrix(NA, nrow = N, ncol = nT + 1)
  logLikelihood <- 0
  # covmats <- list()
  
  # Initialise
  weights[, 1] <- normalisedW[, 1] <- 1/N
  indices[, 1] <- 1:N
  
  filteredState <- c()
  filteredState[1] <- iniState
  particles[, 1] <- iniState
  
  phi <- param$phi
  sigma_v <- param$sigma_v
  sigma_e <- param$sigma_e
  
  for (t in 2:nT) {
    # (i) Resample
    newIndices <- sample(N, replace = TRUE, prob = normalisedW[, t-1])
    
    # particles[, t] <- particles[newIndices, t-1]
    # filteredState[, t] <- filteredState[newIndices, t-1]
    indices[, t] <- newIndices
    
    # (ii) Propagate
    particles[, t] <- phi * particles[newIndices, t-1] + rnorm(N, 0, sigma_v) 
    
    # (iii) Compute weight
    pseudo_xi <- y[t+1] - particles[, t] 
    
    # weights[, t] <- dnorm(y[t+1], mean = pseudo_y, sd = rep(sigma_e, N),
    #                       log = TRUE)
    weights[, t] <- log(dchisq(exp(pseudo_xi), 1) * exp(pseudo_xi)) # log p(y |)
    
    # (iv) Normalise weight
    maxWeight <- max(weights[, t])
    expWeights <- exp(weights[, t] - maxWeight) #exp(weights - maxWeights)
    normalisedW[, t] <- expWeights / sum(expWeights) 
    
    # (v) Compute likelihood
    predictiveLike <- maxWeight + log(sum(expWeights)) - log(N)
    logLikelihood <- logLikelihood + predictiveLike
    
    # (vi) Estimate state
    filteredState[t] <- mean(particles[, t])
    # covmats[[t]] <- 1/(N - 1) * tcrossprod(particles[, t] - mean(particles[, t]))
    
  }
  
  # Return output
  pf_output <- list(filteredState = filteredState, 
                    logLikelihood = logLikelihood)
  
  return(pf_output)
  
}