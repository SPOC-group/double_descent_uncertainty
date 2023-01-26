
import numpy as np

import gcmpyo3

from core.amp.likelihood.base_likelihood import BaseLikelihood
from core.utility import LOGISTIC_APPROX_COEF

class BOLogitLikelihood(BaseLikelihood):
    def __init__(self, noise_var = 0.0) -> None:
        super().__init__()
        self.likelihood = gcmpyo3.Logit(noise_variance = noise_var)
        # self.likelihood = gcmpyo3.NormalizedPseudoBayesLogistic(beta = 1.0 / np.sqrt(1.0 + LOGISTIC_APPROX_COEF**2 * noise_var))

    def Z0(self, y : float, w : int, V : float) -> float:
        return self.likelihood.call_z0(y, w, V)

    def dwZ0(self, y : float, w : int, V : float) -> float:
        return self.likelihood.call_dz0(y, w, V)

    def ddwZ0(self, y : float, w : int, V : float, Z0 : float = None) -> float:
        return self.likelihood.call_ddz0(y, w, V)

    def fout(self, y_list, w_list, V_list):
        return np.real([self.dwZ0(y, w, v) / self.Z0(y, w, v) for (y, w, v) in zip(y_list, w_list, V_list)])

    def dwfout(self, y_list, w_list, V_list):
        return np.real([(self.ddwZ0(y, w, v) / self.Z0(y, w, v)) - (self.dwZ0(y, w, v) / self.Z0(y, w, v))**2 for (y, w, v) in zip(y_list, w_list, V_list)])

    def channel(self, y_list, w_list, V_list) -> float:
        g, dg = self.fout(y_list, w_list, V_list), self.dwfout(y_list, w_list, V_list)
        return g, dg