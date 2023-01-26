"""
TODO : For state evolution : include the computation of the gaussian integral in the prior so that the state evolution only 
has to call the integral function and we can do simplification at the level of the prior
"""

import numpy as np
from scipy.linalg import sqrtm

class NonSepGaussianPrior:
    """
    ASSUMING THAT THE DATA HAS IDENTITY COVARIANCE
    
    """
    def __init__(self, covariance) -> None:
        # for ERM penalization
        self.cov = covariance

    def fa(self, Sigma : float, R : float) -> float:
        """
        Input function, independent of the variance of gaussian prior
        NOTE : Should not depend on the noise in label
        """
        Sigma_mat = np.diag(Sigma)
        # use Woodbury identity : (A^-1 + B^-1)^-1 = A - A (A + B)^-1 A
        inv_sum_inv = Sigma_mat - Sigma_mat @ np.linalg.inv(self.cov + Sigma_mat) @ Sigma_mat
        # a bit complicated but should work
        retour = inv_sum_inv @ (R / Sigma) 
        return retour

    def fv(self, Sigma : float, R : float) -> float:
        """
        Derivative of input function w.r.t. R, multiplied by Sigma
        """
        Sigma_mat = np.diag(Sigma)
        inv_sum_inv = Sigma_mat - Sigma_mat @ np.linalg.inv(self.cov + Sigma_mat) @ Sigma_mat
        # TODO : Verifier si c'est bien comme ca qu'il faut calculer la variance
        return  np.diag(inv_sum_inv)

    def prior(self, b : float, A : float):
        '''
        Compute f and f' for Bernoulli-Gaussian prior
        
        Sigma = 1 / A
        R = b / A
        '''
        retour = self.fa(1. / A, b / A), self.fv(1. / A, b / A)
        return retour
