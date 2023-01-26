import numpy as np
from scipy.integrate import quad
import scipy.stats as stats

from core import utility
from core.amp.likelihood.base_likelihood import BaseLikelihood

import gcmpyo3

class LogitLikelihood(BaseLikelihood):
    """
    Sert a faire du pseudo-Bayesien / Finite temperature logistique
    """
    def __init__(self, beta, normalized = False) -> None:
        super().__init__()
        self.beta    = beta

        if normalized:
            self.likelilhood = gcmpyo3.NormalizedPseudoBayesLogistic(beta = self.beta)
        else:
            self.likelilhood = gcmpyo3.PseudoBayesLogistic(beta = self.beta)

    def Z0(self, y : float, w : int, V : float) -> float:
        return self.likelilhood.call_z0(y, w, V)

    def dwZ0(self, y : float, w : int, V : float) -> float:
        return self.likelilhood.call_dz0(y, w, V)

    def ddwZ0(self, y : float, w : int, V : float, Z0 : float = None) -> float:
        return self.likelilhood.call_ddz0(y, w, V)

    def fout(self, y_list, w_list, V_list):
        return np.real([self.dwZ0(y, w, v) / self.Z0(y, w, v) for (y, w, v) in zip(y_list, w_list, V_list)])

    def dwfout(self, y_list, w_list, V_list):
        return np.real([(self.ddwZ0(y, w, v) / self.Z0(y, w, v)) - (self.dwZ0(y, w, v) / self.Z0(y, w, v))**2 for (y, w, v) in zip(y_list, w_list, V_list)])

    def channel(self, y_list, w_list, V_list) -> float:
        g, dg = self.fout(y_list, w_list, V_list), self.dwfout(y_list, w_list, V_list)
        return g, dg