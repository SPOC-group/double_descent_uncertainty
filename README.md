## Companion repository for the paper "A study of uncertainty quantification in overparametrized high-dimensional models"

This repository contains the code required to reproduce all the plots in the paper, and produce similar plots with different parameters.
It containts four parts : 

- The `gcmpyo3` folder contains the library that computes the state evolution equation for the different estimators. Run `install.sh` to compile the library and generate the Python bindings.

- The `core` folder containts useful functions (e.g. compute the test error and calibration of the estimators, compute temperature scaling, ...)

- The `main.ipynb` produces the plots

- The script `search_lambda.sh` is used to compute the optimal penalizations $\lambda_{\rm error}$, $\lambda_{\rm loss}$ and $\lambda_{\rm evidence}$ for the ERM and empirical Bayes estimators.

### Prerequisites 

The `environment.yml` file contains the required Python packages. You will also need the Rust toolchain to compile the library.
