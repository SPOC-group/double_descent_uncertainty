from numba import jit

@jit
def fa(Sigma, R, lambda_):
    return R / (lambda_ * Sigma + 1.)

@jit
def fv(Sigma, R, lambda_):
    return Sigma / (lambda_ * Sigma + 1.)

class GaussianPrior:
    def __init__(self, lambda_ = 1.0) -> None:
        # for ERM penalization
        self.lambda_ = lambda_
    
    def fa(self, Sigma : float, R : float) -> float:
        """
        Input function, independent of the variance of gaussian prior
        NOTE : Should not depend on the noise in label
        """
        return fa(Sigma, R, self.lambda_)

    def fv(self, Sigma : float, R : float) -> float:
        """
        Derivative of input function w.r.t. R, multiplied by Sigma
        """
        return fv(Sigma, R, self.lambda_)

    def prior(self, b : float, A : float):
        '''
        Compute f and f' for Bernoulli-Gaussian prior
        
        Sigma = 1 / A
        R = b / A
        '''
        return self.fa(1. / A, b / A), self.fv(1. / A, b / A)
