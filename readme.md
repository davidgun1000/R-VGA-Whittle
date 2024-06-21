# R-VGA-Whittle

This folder contains R code for the 5 examples in the R-VGA-Whittle manuscript: 
1. the linear Gaussian state space model (SSM) with simulated data, 
2. the univariate stochastic volatility (SV) model with simulated data,
3. the univariate SV model applied to exchange rate data,  
4. the bivariate SV model with simulated data, and
5. the bivariate SV model applied to exchange rate data.

## Folder structure
Each example is stored in one folder, which contains a `*_main.R` file for running the R-VGA-Whittle, HMC-Whittle and HMC-exact algorithms on the example being considered, and a `*_post_process.R` file for processing the results and generating the posterior density plots. To reproduce the results in the manuscript, run the `*_post_process.R` file.

Each folder contains separate sub-folders:
1. `source`, which contains associated R scripts needed for running R-VGA-Whittle, HMC-Whittle and HMC-exact algorithms on the model considered,
2. `results`, which contains the R-VGA-Whittle, HMC-Whittle and HMC-exact output from the source code 
3. `plots`, which contains the plots of the posterior densities and bivariate posterior plots for each parameter in the model considered,
4. `var_test`, which contains the R code, output and plot for the test on the variance of the R-VGAL results for different Monte Carlo sample sizes (see Section S4 of the online supplement)

## Running the scripts
To reproduce the results in the manuscript, for example that of the linear Gaussian SSM, download the `01_LGSS` folder and run the `lgss_post_process.R` file. Note that the working directory needs to be set to the `01_LGSS` folder for the filepaths to work properly (and similarly, for other examples, set the working directory to the folder containing that example). 

Results from other examples can be similarly reproduced by running the `*_main.R` file in each example's respective folder.

Results from Section S3 of the online supplement can be reproduced by running the `compare_blocksizes.R` file. The flag `rerun_test` is set to `FALSE` by default, so that pre-saved results will be loaded directly from the `results\var_test` sub-folder. Setting this flag to `TRUE` will re-run the R-VGA-Whittle algorithm for the block sizes specified in the `blocksizes` vector (which defaults to 0, 10, 50, 100, 300, 500, and 1000, where 0 denotes no block updating).

Results from Section S4 of the online supplement can be reproduced by running the `var_test_*.R` file. The flag `rerun_rvgaw` is set to `FALSE` by default, so that pre-saved results will be loaded directly from the `var_test` sub-folder. Setting this flag to `TRUE` will re-run the R-VGA-Whittle algorithm 10 times for the Monte Carlo sample size specified in the parameter `S` (set to 1000 by default). 

The RStudio version and R packages required to run the code, along with installation instructions for these packages, can be found in the next section. 

## RStudio version requirements
[]: # (In order to run the HMC code, which was implemented in RStan 2.21, it is highly recommended that you install R version 4.0 or 4.1. The latest released version of RStan at the time of writing is 2.21, which is not yet compatible with R 4.2 and above. There is an RStan development version, 2.26.x, which can be configured to work with R 4.2, but the code in this repository has not been tested on such a configuration.)

In order to run the code, it is highly recommended that you install R version 4.1 or above. The code in this repository was written using R version 4.1 and RStan version 2.21. This code has also been tested and found compatible with R version 4.3 and RStan version 2.26, which are the latest available versions of R and RStan at the time of writing.

Note that prior to installing RStan, you need to configure your R installation to be able to compile C++ code. For instructions, see [RStan Getting Started](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started) under **Configuring C++ Toolchain**. Note that instructions vary depending on your operating system, and if you are using Windows, instructions will also vary depending on your R version (3.6/4.0/4.2). 

The R-VGAL code requires the R package `tensorflow`. It is recommended that you install `tensorflow` version 2.14 or above. First, install the `tensorflow` R package as follows:

```
install.packages("tensorflow")
```
Next, run the following lines of code:
```
library(tensorflow)
install_tensorflow(version = "2.14")
```
which will install `tensorflow` v2.14. If prompted to install Miniconda, select yes by typing 'Y'.

System requirements and a more detailed installation guide for `tensorflow` in R can be found [here](https://tensorflow.rstudio.com/install). 

## Package requirements 
Running the source code requires the following packages (along with their dependencies, which should be installed automatically):
1. `tensorflow` v2.14
2. `rstan` v2.26.23 (for instructions on how to install RStan, see [RStan Getting Started](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started))
3. `ggplot2` v3.4.4
4. `gridExtra` v2.3
5. `gtable` v0.3.4         
6. `mvtnorm` v1.2
7. `dplyr` v1.1.4
