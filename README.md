# double_descent_uncertainty
Companion repository for the paper "On double-descent in uncertainty quantification in overparametrized models"

## Prerequisite 

TODO : Create a bash file (install.sh or sthg like that) to install gcmpyo3
TODO : List the packages required to run everything 

Running the experiments requires the package `gcmpyo3` to be install, in order to run the state evolution equations. As the name implies, it requires the rust toolchain and `pyo3` to be installed.

## Reproducing the figures

The notebook `main.ipynb` can be used to reproduce the figures of the paper, and also produce similar plots in new settings.
To compute the optimal penalizations $\lambda_{\rm error}$, $\lambda_{\rm loss}$ and $\lambda_{\rm evidence}$, use the bash script 
`search_lambda.sh`. 

