## Companion repository for the paper "A study of uncertainty quantification in overparametrized high-dimensional models" ( https://arxiv.org/abs/2210.12760 )

This repository contains the code required to reproduce the plots in the paper.
It containts four parts : 

- The `gcmpyo3` folder contains the library that computes the state evolution equation for the different estimators. Run `install.sh` to compile the library and generate the Python bindings.

- The `core` folder containts useful functions (e.g. compute the test error and calibration of the estimators, compute temperature scaling, ...)

- The `main.ipynb` produces the theoretical curves using the state evolution equations described in the main part of the paper.

- The `experiments` folder contains one notebook for each estimator (bayes optimal, erm, pseudo-Bayes)

## Prerequisites 

The `environment.yml` file contains the required Python packages. You will also need the Rust toolchain to compile the library contained in the `gcmpyo3` folder.
