find_cutoff_freq <- function(signal, nsegs, power_prop = 0.5, save_plots = FALSE) {

    y <- signal
    ## Periodogram smoothing
    pdg_out <- compute_periodogram(y)
    freq <- pdg_out$freq
    pdg <- pdg_out$periodogram

    welch_output <- compute_welch_psd(y, nperseg = length(y)/nsegs, p_overlap = 0.5)
    freq_welch <- welch_output$freq
    pdg_welch <- welch_output$pdg

    # plot(freq, pdg_original, type = "l")
    # lines(freq_welch, pdg_welch, col = "red")

    # dB <- 10 * log10(pdg_welch)
    # half_power <- 10*log10(max(pdg_welch) * power_prop)
    # beyond_half <- which(dB >= half_power)#[1]

    half_power <- max(pdg_welch) * power_prop
    beyond_half <- which(pdg_welch >= half_power)
    cutoff <- beyond_half[length(beyond_half)] # the 3dB cutoff is the last frequency bin with above half power

    cutoff_freq <- freq_welch[cutoff]
    cutoff_freq_og <- freq[freq >= cutoff_freq][1]
    cutoff_ind_og <- which(freq == cutoff_freq_og)

    # plot(freq, pdg_og, col = "grey", type = "l", 
    #     main = paste0("Cutoff at the ", cutoff_ind_og, "th frequency"), 
    #     xlab = "Frequency", ylab = "Periodogram")
    # lines(freq_welch, pdg_welch, col = "black", lwd = 2)
    # abline(v = cutoff_freq_og, lty = 2, lwd = 2, col = "red")

    pdg_df <- data.frame(freq = freq, pdg = pdg)
    pdg_welch_df <- data.frame(freq = freq_welch, pdg = pdg_welch)

    pdg_plot <- pdg_df %>% ggplot(aes(x = freq, y = pdg)) +
    geom_line() +
    geom_line(data = pdg_welch_df, aes(x = freq, y = pdg), 
                color = "salmon", linewidth = 1.5) +
    geom_vline(xintercept = cutoff_freq_og, 
                linetype = 2, color = "red", linewidth = 1) +
    labs(x = "Frequency (rad/s)", y = "Power") +
    xlim(c(0, 1)) +
    theme_bw() +
    theme(text = element_text(size = 24))

    print(pdg_plot)

    return(list(pdg_welch = pdg_welch,
                freq_welch = freq_welch,
                cutoff_ind = cutoff_ind_og, 
                cutoff_freq = cutoff_freq_og, 
                pdg_plot = pdg_plot))
} 
  
 
 compute_welch_psd <- function(signal, window='hamming', 
                                nperseg=NULL, p_overlap=NULL) {
    # Computes the Power Spectral Density (PSD) of a signal using Welch's method.

    # Parameters:
    # - signal : array_like
    #     Input signal data.
    # - fs : float, optional
    #     Sampling frequency of the signal. Default is 1.0.
    # - window : str or tuple or array_like, optional
    #     Desired window to use. Can be a string (e.g., 'hann' for Hann window), tuple, or array. Default is 'hann'.
    # - nperseg : int, optional
    #     Length of each segment. Default is None, which sets it to 256.
    # - noverlap : int, optional
    #     Number of points to overlap between segments. Default is None, which sets it to nperseg // 2.

    # Returns:
    # - f : ndarray
    #     Array of sample frequencies.
    # - psd : ndarray
    #     Power spectral density of signal.

    # Use default segment length if not specified
    if (is.null(nperseg)) {
        nperseg <- 250 #256  # Reasonable default
    }
    # Default overlap is half the segment size
    noverlap <- 0
    if (is.null(noverlap)) {
        noverlap <- nperseg * 0.5
    } else {
        noverlap <- nperseg * p_overlap
    }

    # Calculate the windowing function
    window = get_window(window, nperseg)

    # Divide signals into segs
    M <- length(signal)/nperseg
    start = seq(1, length(signal) - noverlap, nperseg - noverlap)
    # end = start + nperseg - 1
    # inds <- lapply(1:M, function(col) rbind(start, end)[, col])
    segs <- lapply(start, function(i) signal[i:(i + nperseg - 1)])

    # Compute the PSD using Welch's method
    A <- lapply(segs, function(seg) 1/length(seg) * fft(window * seg))
    U <- 1/nperseg * sum(window^2)
    
    periodogram_seg <- lapply(A, function(seg) nperseg / U * Mod(seg)^2)

    # Compute the mean of periodogram_seg
    periodogram <- rowMeans(do.call(cbind, periodogram_seg))
    # half_pdg <- periodogram[1:(length(periodogram)/2)]
    # freq <- (1:(nperseg/2)) / nperseg * (2*pi)

    # Take only the first half of the periodogram
    k <- 1:floor((nperseg-1)/2)
    half_pdg <- periodogram[k + 1] # +1 because fft() includes the zero frequency

    freq <- k / nperseg * (2*pi)
    half_freq <- freq[k]

    # Compute the PSD using Welch's method
    # psd = welch(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)

    return(list(freq = freq, pdg = half_pdg))
}

get_window <- function(window, N) { # might want to try a Hamming window t
    # Generate a Hann window.

    # Parameters:
    # - N : int
    #     The number of window points.

    # Returns:
    # - window : ndarray
    #     The Hann window.
    
    if (window == "hann") {
        n <- 1:N
        window <- 0.5 - 0.5 * cos(2 * pi * n / (N - 1))
    } else if (window == "hamming") {
        n <- 1:N
        window <- 25/46 - (1 - 25/46) * cos(2 * pi * n / (N - 1))
    } else {
       window <- rep(1, N)
    }
    return(window)
}

# test <- rnorm(10000)
# psd <- compute_welch_psd(test, fs=1.0, window='hann', nperseg=500, p_overlap=0.5)
# plot(psd)
