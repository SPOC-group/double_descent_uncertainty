# double_descent_uncertainty
Companion repository for the paper "A study of uncertainty quatification in overparametrized high-dimensional models"

## Prerequisite 

TODO : List the packages required to run everything 

Running the experiments requires the package `gcmpyo3` to be installed. Since it is made of Python bindings of a Rust library, it requires the rust toolchain and the packages `pyo3` and `maturin` to be installed.

## Reproducing the figures

The notebook `main.ipynb` can be used to reproduce the figures of the paper, and also produce similar plots in new settings.
To compute the optimal penalizations $\lambda_{\rm error}$, $\lambda_{\rm loss}$ and $\lambda_{\rm evidence}$, use the bash script 
`search_lambda.sh` and modify the parameters. 

