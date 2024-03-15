from tabnanny import verbose
from time import time
from typing import overload

import gcmpyo3

import numpy           as np
import scipy.optimize  as optimize
import gcmpyo3

from core import utility

# NOTE : n = samples, p = parameters = dimension of student, d = dimension of teacher

def empirical_bayes_compute_optimal_lambda_for_evidence(n_over_p, p_over_d, noise_std, matching, student_activation, rho = 1.0, lambda_min = 1e-3, lambda_max = 10.0, data_model = "logit"):
    """
    student_activation : only useful is matching is False
    """
    student_activation = student_activation if (not matching) else 'matching'
    se_tolerance       = 1e-4
    # the value of beta does not matter
    beta = 1.0

    if matching:
        def to_minimize(lambda_):
            m, q, v, mhat, qhat, vhat = gcmpyo3.state_evolution.pseudo_bayes_state_evolution_matching(n_over_p, beta, noise_std**2, lambda_, rho, data_model, se_tolerance, False, normalized = True, verbose = False)
            return - gcmpyo3.evidence.pseudo_bayes_log_evidence_matching(m, q, v, mhat, qhat, vhat, n_over_p, beta, noise_std**2, lambda_, rho, data_model)
    
    else:
        _, kappa1, kappastar = utility.KERNEL_COEFICIENTS[student_activation]
        def to_minimize(lambda_):
            m, q, v, mhat, qhat, vhat = gcmpyo3.state_evolution.pseudo_bayes_state_evolution_gcm(n_over_p, beta, noise_std**2, p_over_d, kappa1, kappastar, lambda_, rho, data_model, se_tolerance, False, normalized = True, verbose = False)
            return - gcmpyo3.evidence.pseudo_bayes_log_evidence_gcm(m, q, v, mhat, qhat, vhat, n_over_p, beta, noise_std**2, p_over_d, kappa1, kappastar, lambda_, rho, data_model)

    opt_res = optimize.minimize_scalar(to_minimize,
                                       method = 'bounded',
                                       bounds =[lambda_min, lambda_max])
    lambda_opt = opt_res.x
    return lambda_opt

def empirical_bayes_compute_optimal_lambda_for_error(n_over_p, p_over_d, noise_std, matching, student_activation, rho = 1.0, lambda_min = 1e-3, lambda_max = 10.0, data_model = "logit"):
    se_tolerance = 1e-4
    beta         = 1.0

    if matching:
        def to_minimize(lambda_):
            m, q, v, mhat, qhat, vhat = gcmpyo3.state_evolution.pseudo_bayes_state_evolution_matching(n_over_p, beta, noise_std**2, lambda_, rho, data_model, se_tolerance, True, normalized = True, verbose = False)
            error = - m / np.sqrt(q)
            return error
    else:
        _, kappa1, kappastar = utility.KERNEL_COEFICIENTS[student_activation]
        
        def to_minimize(lambda_):
            m, q, v, mhat, qhat, vhat = gcmpyo3.state_evolution.pseudo_bayes_state_evolution_gcm(n_over_p, 1.0, noise_std**2, p_over_d, kappa1, kappastar, lambda_, rho, data_model, se_tolerance, False, normalized = True, verbose = False)
            error = - m / np.sqrt(q)
            return error
    
    opt_res = optimize.minimize_scalar(lambda x : to_minimize(x),
                                method = 'bounded',
                                bounds =[lambda_min, lambda_max])
    return opt_res.x

##### FONCTIONS POUR ERM #####

def erm_compute_optimal_lambda_for_error(n_over_p, noise_std, p_over_d, student_activation, lambda_min, lambda_max, se_tolerance, matching, rho, data_model = "logit"):
    xatol = 1e-4

    if matching:
        def to_minimize(lambda_):
            m, q, _, _, _, _ = gcmpyo3.state_evolution.erm_state_evolution_matching(n_over_p, noise_std**2, lambda_, rho, data_model, se_tolerance, False, False)
            return -m / np.sqrt(q)
    else:
        _, kappa1, kappastar = utility.KERNEL_COEFICIENTS[student_activation]
        
        def to_minimize(lambda_):
            m, q, _, _, _, _ = gcmpyo3.state_evolution.erm_state_evolution_gcm(n_over_p, noise_std**2, p_over_d, kappa1, kappastar, lambda_, rho, data_model, se_tolerance, False)
            return - m / np.sqrt(q)

    opt_res = optimize.minimize_scalar(lambda lambda_ : to_minimize(lambda_),
                                       method = 'bounded',
                                       bounds =[lambda_min, lambda_max],
                                       options={'xatol' : xatol})
    lambda_opt = opt_res.x
    return lambda_opt

def erm_compute_optimal_lambda_for_loss(n_over_p, noise_std, p_over_d, student_activation, lambda_min, lambda_max, se_tolerance, matching, rho, data_model = "logit"):
    xatol      = 1e-4

    loss_function_0 = utility.generalisation_loss_logit_teacher if data_model == "logit" else utility.generalisation_loss_probit_teacher

    if matching:
        def to_minimize(lambda_):
            m, q, _, _, _, _ = gcmpyo3.state_evolution.erm_state_evolution_matching(n_over_p, noise_std**2, lambda_, rho, data_model, se_tolerance, False, False)
            # last argument is teacher variance
            return loss_function_0(rho, m, q, noise_std**2)

    else:
        _, kappa1, kappastar = utility.KERNEL_COEFICIENTS[student_activation]
        additional_variance  = utility.get_additional_noise_from_kappas(kappa1, kappastar, p_over_d)
        # last argument is teacher variance
        loss_function = lambda rho, m, q : loss_function_0(rho * (1.0 - additional_variance), m, q, 0.0, noise_std**2 + rho * additional_variance)

        def to_minimize(lambda_):
            m, q, _, _, _, _ = gcmpyo3.state_evolution.erm_state_evolution_gcm(n_over_p, noise_std**2, p_over_d, kappa1, kappastar, lambda_, rho, data_model, se_tolerance, False)
            return loss_function(rho, m, q)

    opt_res = optimize.minimize_scalar(lambda lambda_ : to_minimize(lambda_),
                                       method = 'bounded',
                                       bounds =[lambda_min, lambda_max],
                                       options= {'xatol' : xatol} )
    lambda_opt = opt_res.x
    return lambda_opt
