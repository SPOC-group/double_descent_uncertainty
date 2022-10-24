import scipy.optimize
import numpy as np

try:
    import utility
except:
    import core.utility as utility

def find_optimal_temperature_logit_teacher(rho, m, q, teacher_variance, t_mini = 0.01, t_maxi = 10.0):
    # For now, focus on logistic regression => student noise is 0
    def aux(temp):
        m_tilde, q_tilde = m / temp, q / temp**2
        return utility.generalisation_loss_logit_teacher(rho, m_tilde, q_tilde, 0.0, teacher_variance)

    res = scipy.optimize.minimize_scalar(aux, method='bounded', bounds = [t_mini, t_maxi])
    return res.x

def find_optimal_temperature_probit_teacher(rho, m, q, teacher_variance, t_mini = 0.01, t_maxi = 10.0):
    # For now, focus on logistic regression => student noise is 0
    def aux(temp):
        m_tilde, q_tilde = m / temp, q / temp**2
        return utility.generalisation_loss_probit_teacher(rho, m_tilde, q_tilde, 0.0, teacher_variance)

    res = scipy.optimize.minimize_scalar(aux, method='bounded', bounds = [t_mini, t_maxi])
    return res.x
