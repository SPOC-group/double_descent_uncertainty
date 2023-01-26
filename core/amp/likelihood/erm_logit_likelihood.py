from lib2to3.pytree import Base
from base_likelihood import BaseLikelihood
import numpy as np
from ....core import utility

class ERMLogitLikelihood(BaseLikelihood):
    def __init__(self) -> None:
        super().__init__()

    def fout(self, w, y, V):
        logistic = lambda x : np.log(1. + np.exp(-y*x))
        logistic_prime = lambda x : - y / (1. + np.exp(y * x))
        logistic_second = lambda x : - (y**2) * np.exp(y  * x) / (1. + np.exp(y * x))**2
        # should be correct
        prox = utility.proximal_operator(logistic, w, V)
        return (1. / V) * (prox - w)

    def dwfout(self, w, y, V):
        # do not recompute the proximal operator twice, reuse previous computations
        f = f or self.fout(w, y, V)
        # On peut enlever le y du cosh par symmetrie de la fonction
        alpha = (2. * np.cosh(0.5 * y * (w + V*f)))**2
        # Sanity check : apparement, pour que le onsager tBaseERM soit globalement positif, il faut que dwgout soit negatif
        return - 1. / (alpha + V)

    def channel(self, y, w, V):
        n = len(w)
        g, dg = np.zeros_like(y), np.zeros_like(y)
        for i in range(n):
            g[i] = self.fout(w[i], y[i], V[i])
            dg[i] = self.dwfout(w[i], y[i], V[i], f = g[i])
        return g, dg

    # EQUATIONS FOR STATE EVOLUTION 