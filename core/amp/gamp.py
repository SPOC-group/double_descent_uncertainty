
from cmath import nan
from code import InteractiveInterpreter
from typing import List
import numpy as np

from ..amp.prior.gaussian_prior import GaussianPrior
from ..amp.likelihood.base_likelihood import BaseLikelihood
from ..utility import *

def iterate_gamp(X : List[List[float]], Y : List[float], w0 : List[float], likelihood, prior, max_iter : int = 200, tol : float =1e-7, 
                 damp : float =0.0, early_stopping : bool =False, verbose : bool = False) -> dict:
    """
    MAIN FUNCTION : Runs G-AMP and returns the finals parameters. If we study
    the variance, we are interested in the vhat quantities. The 'variance' of the vector 
    w will (normally) be the sum of the vhat.

    parameters :
        - W : data matrix
        - y : funciton output
        - w0 : ground truth
    returns : 
        - retour : dictionnary with informations
    """
    d = len(w0)

    # Preprocessing
    y_size, x_size = X.shape
    X2 = X * X
    
    # Initialisation
    xhat = np.zeros(x_size)
    vhat = np.ones(x_size)
    g = np.zeros(y_size)

    count = 0

    status = None

    for t in range(max_iter):
        # First part: m-dimensional variables
        V     = X2 @ vhat
        # here we see that V is the Onsager term
        omega = X @ xhat - V * g
        g, dg = likelihood.channel(Y, omega, V)

        # Second part
        A = -X2.T @ dg
        b = A*xhat + X.T @ g
        
        xhat_old = xhat.copy() # Keep a copy of xhat to compute diff

        xhat, vhat = prior.prior(b, A)

        diff = np.mean(np.abs(xhat-xhat_old))
        # Expression of MSE has been changed

        if (diff < tol):
            status = 'Done'
            break

        if verbose:
            q = np.mean(xhat * xhat)
            print(f'q = {q}')
            print(f'Variance = {np.mean(vhat)}')

    if verbose:
        print('t : ', t)
        print(f'diff : {diff}')

    retour = {}
    retour['estimator'] = xhat
    retour['variances'] = vhat
    
    return retour

def get_bayes_overlap_from_gamp(w0, what, vhat, Omega_inv_sqrt, Phi):
    # normally : p is the teacher 
    p, d = Phi.shape
    V = np.sum(vhat) / p
    q = (what @ what) / p
    m =  (w0 @ Phi @ Omega_inv_sqrt @ what) / p
    return {'V' : V, 'q' : q, 'm' : m, 'rho' : np.trace(Omega_inv_sqrt @ Phi.T @ Phi @ Omega_inv_sqrt) / p}