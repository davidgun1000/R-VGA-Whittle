run_rvgaw_lgss <- function(y, phi = NULL, sigma_eta = NULL, sigma_eps = NULL,
                           transform = "arctanh",
                           prior_mean = 0, prior_var = 1,
                           deriv = "tf", S = 1000L,
                           use_tempering = T, temper_schedule = rep(1 / 10, 10),
                           n_temper = 100,
                           temper_first = T,
                           reorder, reorder_seed = 2024,
                           n_reorder = NULL,
                           decreasing = F,
                           n_indiv = NULL,
                           nblocks = NULL) {
    print("Starting R-VGA with Whittle likelihood...")

    rvgaw.t1 <- proc.time()

    n <- length(y)

    rvgaw.mu_vals <- list()
    rvgaw.mu_vals[[1]] <- prior_mean

    rvgaw.prec <- list()
    rvgaw.prec[[1]] <- chol2inv(chol(prior_var))

    ## Fourier frequencies
    # k <- seq(-ceiling(n/2)+1, floor(n/2), 1)
    # k_in_likelihood <- k[k >= 1 & k <= floor((n-1)/2)]
    # freq <- 2 * pi * k_in_likelihood / n

    # ## Fourier transform of the observations
    # fourier_transf <- fft(y)
    # periodogram <- 1/n * Mod(fourier_transf)^2
    # I <- periodogram[k_in_likelihood + 1]
    pgram_output <- compute_periodogram(y)
    freq <- pgram_output$freq
    I <- pgram_output$periodogram

    if (reorder == "decreasing") {
        sorted_freq <- sort(freq, decreasing = T, index.return = T)
        indices <- sorted_freq$ix
        reordered_freq <- sorted_freq$x
        reordered_I <- I[indices]
    } else if (reorder == "random") {
        set.seed(reorder_seed)
        indices <- sample(1:length(freq), length(freq))
        reordered_freq <- freq[indices]
        reordered_I <- I[indices]
    } else if (reorder > 0) {
        n_reorder <- reorder
        # n_others <- length(freq - n_reorder)
        original <- (n_reorder + 1):length(freq)
        inds <- ceiling(seq(n_reorder, length(freq) - n_reorder, length.out = n_reorder))
        new <- original
        for (j in 1:n_reorder) {
            new <- append(new, j, after = inds[j])
        }
        reordered_freq <- freq[new]
        reordered_I <- I[new]
    } else { ## do nothing
        reordered_freq <- freq
        reordered_I <- I
    }

    freq <- reordered_freq
    I <- reordered_I
    # browser()
    # LB <- c()

    if (use_tempering) {
        if (temper_first) {
            cat("Damping the first ", n_temper, "frequencies... \n")
        } else {
            cat("Damping the last ", n_temper, "frequencies... \n")
        }
    }

    all_blocks <- as.list(1:length(freq))

    if (!is.null(nblocks)) {
        # Split frequencies into blocks
        # Last block may not have the same size as the rest
        # if the number of frequencies to be divided into blocks
        # is not divisible by the number of blocks
        indiv <- list()
        vec <- c()
        if (reorder == 0) { # leave the first n_indiv frequencies alone, cut the rest into blocks
            
            if (n_indiv == 0) {
                vec <- (n_indiv+1):length(freq)
                blocks <- split(vec, cut(seq_along(vec), nblocks, labels = FALSE))
                all_blocks <- blocks
            } else {
                indiv <- as.list(1:n_indiv)
                vec <- (n_indiv+1):length(freq)
                blocks <- split(vec, cut(seq_along(vec), nblocks, labels = FALSE))
                all_blocks <- c(indiv, blocks)    
            }
             
        } else if (reorder == "decreasing") { # leave the last n_indiv frequencies alone, cut the rest into blocks
            indiv <- as.list((length(freq) - n_indiv):length(freq))
            vec <- 1:(length(freq) - n_indiv)
            blocks <- split(vec, cut(seq_along(vec), nblocks, labels = FALSE))
            all_blocks <- c(blocks, indiv)
        }
    }
    
    n_updates <- length(all_blocks)
    for (i in 1:n_updates) {
        cat("i =", i, "\n")
        blockinds <- all_blocks[[i]]

        a_vals <- 1
        if (use_tempering) {
            if (temper_first) {
                if (i <= n_temper) { # only temper the first n_temper observations
                    a_vals <- temper_schedule
                }
            } else {
                if (i > length(freq) - n_temper) { # only temper the first n_temper observations

                    a_vals <- temper_schedule
                }
            }
        }

        mu_temp <- rvgaw.mu_vals[[i]]
        prec_temp <- rvgaw.prec[[i]]

        log_likelihood <- c()

        for (v in 1:length(a_vals)) { # for each step in the tempering schedule

            a <- a_vals[v]

            P <- chol2inv(chol(prec_temp))
            samples <- rmvnorm(S, mu_temp, P)
            theta_phi <- samples[, 1]
            theta_eta <- samples[, 2]
            theta_eps <- samples[, 3]

            grads <- list()
            hessian <- list()

            # use Tensorflow to compute the grad and hessian
            tf.t1 <- proc.time()

            samples_tf <- tf$Variable(samples, dtype = "float32")
            # theta_phi_tf <- tf$Variable(theta_phi)
            # theta_eta_tf <- tf$Variable(theta_eta)
            # theta_eps_tf <- tf$Variable(theta_eps)

            freq_i_tf <- tf$constant(freq[blockinds])
            I_i_tf <- tf$constant(I[blockinds])

        
            tf_out_test <- compute_grad_arctanh_test(samples_tf, I_i_tf, freq_i_tf,
                                                     blocksize = length(blockinds))
            E_grad_tf <- tf_out_test$E_grad
            E_hessian_tf <- tf_out_test$E_hessian

            tf.t2 <- proc.time()

            E_grad <- as.array(E_grad_tf)
            E_hessian <- as.array(E_hessian_tf)

            ## Update variational mean and precision

            prec_temp <- prec_temp - a * E_hessian

            eigvals <- eigen(prec_temp)$value
            if (any(eigvals < 0)) {
                # browser() ## try nearPD() funciton from the Matrix package here
                neg_eigval <- eigvals[eigvals < 0]
                cat("Warning: precision matrix has negative eigenvalue", neg_eigval, "\n")
                prec_temp <- as.matrix(nearPD(prec_temp)$mat)
                browser()
            }

            mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * E_grad)
        }

        rvgaw.prec[[i + 1]] <- prec_temp
        rvgaw.mu_vals[[i + 1]] <- mu_temp

        if (i %% floor(n_updates / 10) == 0) {
            cat(floor(i / n_updates * 100), "% complete \n")
        }
    }

    rvgaw.t2 <- proc.time()

    ## Posterior samples
    rvgaw.post_var <- chol2inv(chol(rvgaw.prec[[n_updates+1]]))

    theta.post_samples <- rmvnorm(10000, rvgaw.mu_vals[[n_updates+1]], rvgaw.post_var) # these are samples of beta, log(sigma_a^2), log(sigma_e^2)

    rvgaw.post_samples_phi <- 0
    if (transform == "arctanh") {
        rvgaw.post_samples_phi <- tanh(theta.post_samples[, 1])
    } else {
        rvgaw.post_samples_phi <- exp(theta.post_samples[, 1]) / (1 + exp(theta.post_samples[, 1]))
    }
    rvgaw.post_samples_eta <- sqrt(exp(theta.post_samples[, 2]))
    rvgaw.post_samples_eps <- sqrt(exp(theta.post_samples[, 3]))

    rvgaw.post_samples <- list(
        phi = rvgaw.post_samples_phi,
        sigma_eta = rvgaw.post_samples_eta,
        sigma_eps = rvgaw.post_samples_eps
    )

    # plot(density(rvgaw.post_samples))

    ## Save results
    rvgaw_results <- list(
        mu = rvgaw.mu_vals,
        prec = rvgaw.prec,
        post_samples = rvgaw.post_samples,
        transform = transform,
        S = S,
        use_tempering = use_tempering,
        temper_schedule = a_vals,
        # lower_bound = LB,
        time_elapsed = rvgaw.t2 - rvgaw.t1
    )

    return(rvgaw_results)
}

log_likelihood_arctanh <- function(theta_phi, theta_eta, theta_eps,
                                   omega_k, I_k) {
    phi <- tanh(theta_phi)
    sigma_eta <- sqrt(exp(theta_eta))
    sigma_eps <- sqrt(exp(theta_eps))

    spec_dens_x <- sigma_eta^2 / (1 + phi^2 - 2 * phi * cos(omega_k))
    spec_dens_eps <- sigma_eps^2 # / (2*pi)
    spec_dens_y <- spec_dens_x + spec_dens_eps
    llh <- -log(spec_dens_y) - I_k / spec_dens_y
}


compute_grad_arctanh_test <- tf_function(
    compute_grad_arctanh_test <- function(samples_tf, I_i, freq_i, blocksize) {
        log_likelihood_tf <- 0
        with(tf$GradientTape() %as% tape2, {
            with(tf$GradientTape(persistent = TRUE) %as% tape1, {
                S <- as.integer(nrow(samples_tf))
                
                # nfreq <- as.integer(length(freq_i))
                phi_s <- tf$math$tanh(samples_tf[, 1])
                phi_s <- tf$reshape(phi_s, c(length(phi_s), 1L, 1L)) # S x 1 x 1
                freq_i <- tf$reshape(freq_i, c(1L, blocksize, 1L)) # 1 x blocksize x 1
               
                sigma_eta2_s <- tf$math$exp(samples_tf[, 2])
                sigma_eta2_s <- tf$reshape(sigma_eta2_s, c(dim(sigma_eta2_s), 1L, 1L))
                sigma_eta2_tiled <- tf$tile(sigma_eta2_s, c(1L, blocksize, 1L))
            
                spec_dens_x_tf <- tf$math$divide(sigma_eta2_tiled, 
                                                1 + tf$tile(tf$math$square(phi_s), c(1L, blocksize, 1L)) -
                                                tf$math$multiply(2, tf$math$multiply(phi_s, tf$math$cos(freq_i))))

                ## add spec_dens_eps here
                spec_dens_eps_tf <- tf$math$exp(samples_tf[, 3])
                spec_dens_eps_tf <- tf$reshape(spec_dens_eps_tf, c(S, 1L, 1L))
                ## then
                spec_dens_y_tf <- spec_dens_x_tf + tf$tile(spec_dens_eps_tf, c(1L, blocksize, 1L))

                I_i <- tf$reshape(I_i, c(1L, blocksize, 1L))
                I_tile <- tf$tile(I_i, c(S, 1L, 1L))
                log_likelihood_tf <- -tf$math$log(spec_dens_y_tf) - tf$multiply(I_i, tf$math$reciprocal(spec_dens_y_tf))

                log_likelihood_tf <- tf$math$reduce_sum(log_likelihood_tf, 1L) # sum all log likelihoods over the block
            })
            grad_tf %<-% tape1$gradient(log_likelihood_tf, samples_tf)
            # grad_tf <- tf$reshape(tf$transpose(grad_tf), c(dim(grad_tf[[1]]), 3L))
        })

        grad2_tf %<-% tape2$batch_jacobian(grad_tf, samples_tf)
        # grad2_tf %<-% tape2$batch_jacobian(grad_tf, vars)

        E_grad_tf <- tf$reduce_mean(grad_tf, 0L)
        E_hessian_tf <- tf$reduce_mean(grad2_tf, 0L)

        return(list(
            log_likelihood = log_likelihood_tf,
            grad = grad_tf,
            hessian = grad2_tf,
            E_grad = E_grad_tf,
            E_hessian = E_hessian_tf
        ))
    }
)
