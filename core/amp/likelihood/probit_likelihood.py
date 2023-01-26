
import numpy as np
from scipy.special import erfc
import scipy.stats as stats
from scipy.integrate import quad, nquad
from scipy.stats import norm 

import core.utility as utility
from core.amp.likelihood.base_likelihood import BaseLikelihood

H_ = lambda x : 0.5 * erfc(x / np.sqrt(2.))

class ProbitLikelihood(BaseLikelihood):
    def __init__(self, sigma) -> None:
        self.sigma = sigma
        self.bound = 5.0

    def fout(self, w, y, V):
        delta = 1e-10
        # Only change this part to take the noise into account
        U = V + self.sigma**2
        try:
            deno = np.sqrt(2*np.pi * U) * H_(- y * w / np.sqrt(U)) + delta
            x = y * np.exp(-0.5*(w**2 / U)) / deno
        except Warning:
            print('Error in fout of Bayes')
            return 0
        return x

    def dwfout(self, w, y, V):
        delta = 1e-10
        U = V + self.sigma**2
        g = self.fout(w, y, V)
        tmp = np.multiply(g, (np.divide(w, U + delta) + g))
        return - np.maximum(tmp, 0.)
    
    def channel(self, y : int, w : float, v : float) -> float:
        return self.fout(w, y, v), self.dwfout(w, y, v)
