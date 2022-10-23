# TODO : Add option to maximize evidence
# -> compute state evolution with normalized pseudo bayes 
# -> qty to maximize is the free energy (TODO : implement it ) + 0.5 * log(beta * lambda_)
# TODO : Check it coincides in simple cases (logit, probit data + matching) 

import argparse
import csv
from tabnanny import verbose
from time import time
from typing import overload

import gcmpyo3

import numpy           as np
import scipy.optimize  as optimize
from skopt             import gp_minimize
import gcmpyo3

from core.se.logistic_regression_se import gcm_logistic_kappa_run_se, gcm_matching_logistic_run_se
from core.se.pseudo_bayes_se import pseudo_bayes_matching_run_se, pseudo_bayes_random_features_run_se
from core import data, utility
from core.utility import generalisation_error_probit_teacher, generalisation_loss_logit_teacher, generalisation_loss_probit_teacher

# == Liste des arguments pour le script 

DEFAULT_LAMBDA_MIN         = 1e-4
DEFAULT_LAMBDA_MAX         = 10.0

DEFAULT_BETA_MIN           = 1e-3
DEFAULT_BETA_MAX           = 10.0

DEFAULT_SE_TOLERANCE       = 1e-10
DEFAULT_XATOL              = 1e-10
# TEMPORARY
USE_BAYESIAN_OPT           = True

DEFAULT_FILENAME             = 'experiments_results/search_optimal_lambda.csv'
DEFAULT_FILENAME_LOGISTIC    = 'experiments_results/search_optimal_lambda_logistic_data.csv'
DEFAULT_FILENAME_LAMBDA_BETA = 'experiments_results/search_optimal_beta_lambda.csv'

##### FONCTIONS POUR PSEUDO-BAYES #####

def pseudo_bayesian_compute_optimal_lambda_for_evidence(alpha, sigma, gamma, student_activation, lambda_min, lambda_max, se_tolerance, matching, use_logistic_data, rho):
    data_model = {False : "probit", True : "logit"}[use_logistic_data]
    # the value of beta does not matter
    beta = 1.0

    if matching:
        def to_minimize(lambda_):
            m, q, v, mhat, qhat, vhat = gcmpyo3.state_evolution.pseudo_bayes_state_evolution_matching(alpha, beta, sigma**2, lambda_, rho, data_model, se_tolerance, False, normalized = True, verbose = False)
            return - gcmpyo3.evidence.pseudo_bayes_log_evidence_matching(m, q, v, mhat, qhat, vhat, alpha, beta, sigma**2, lambda_, rho, data_model)
    
    else:
        _, kappa1, kappastar = utility.KERNEL_COEFICIENTS[student_activation]
        def to_minimize(lambda_):
            print(f'Trying lambda = {lambda_}')
            print(f'    Rho = {rho * (1.0 - utility.get_additional_noise_from_kappas(kappa1, kappastar, gamma))}')
            m, q, v, mhat, qhat, vhat = gcmpyo3.state_evolution.pseudo_bayes_state_evolution_gcm(alpha, beta, sigma**2, gamma, kappa1, kappastar, lambda_, rho, data_model, se_tolerance, False, normalized = True, verbose = False)
            print(m, q, v, mhat, qhat, vhat)
            print('     Computin evidence')
            return - gcmpyo3.evidence.pseudo_bayes_log_evidence_gcm(m, q, v, mhat, qhat, vhat, alpha, beta, sigma**2, gamma, kappa1, kappastar, lambda_, rho, data_model)

    opt_res = optimize.minimize_scalar(to_minimize,
                                       method = 'bounded',
                                       bounds =[lambda_min, lambda_max])
    lambda_opt = opt_res.x
    print(f"With beta = {beta}, optimal lambda is {lambda_opt}")
    return lambda_opt

def pseudo_bayesian_compute_optimal_lambda_for_error(alpha, sigma, gamma, student_activation, lambda_min, lambda_max, se_tolerance, matching, use_logistic_data, optimize_beta, beta_min, beta_max, rho):
    data_model = ['probit', 'logit'][int(use_logistic_data)]

    if matching:
        def to_minimize(x):
            beta, lambda_ = x
            m, q, v, mhat, qhat, vhat = gcmpyo3.state_evolution.pseudo_bayes_state_evolution_matching(alpha, beta, sigma**2, lambda_, rho, data_model, se_tolerance, True, normalized = True, verbose = False)
            error = - m / np.sqrt(q)
            return error
    else:
        _, kappa1, kappastar = utility.KERNEL_COEFICIENTS[student_activation]
        
        def to_minimize(x):
            beta, lambda_ = x
            m, q, v, mhat, qhat, vhat = gcmpyo3.state_evolution.pseudo_bayes_state_evolution_gcm(alpha, beta, sigma**2, gamma, kappa1, kappastar, lambda_, rho, data_model, se_tolerance, False, normalized = True, verbose = False)
            error = - m / np.sqrt(q)
            return error
    
    if not optimize_beta:
        # fix beta = 1.0
        opt_res = optimize.minimize_scalar(lambda x : to_minimize([1.0, x]),
                                    method = 'bounded',
                                    bounds =[lambda_min, lambda_max])
        lambda_opt = opt_res.x
        return 1.0, opt_res.x

    if optimize_beta:
        opt_res = optimize.minimize(to_minimize, x0 = [np.random.uniform(beta_min, beta_max), np.random.uniform(lambda_max, lambda_min)], 
        bounds =[(beta_min, beta_max), (lambda_min, lambda_max)], 
        # tol=DEFAULT_XATOL,
        # method='Nelder-Mead'
        )
        beta_opt, lambda_opt = opt_res.x
        
        print('Optimal beta, lambda are', beta_opt, lambda_opt)
        return beta_opt, lambda_opt

# This function is not used at the moment

def pseudo_bayesian_compute_optimal_lambda_for_loss(alpha, sigma, gamma, student_activation, lambda_min, lambda_max, se_tolerance, matching, use_logistic_data, optimize_beta, beta_min, beta_max):
    data_model = ['probit', 'logit'][int(use_logistic_data)]
    if matching:
        def to_minimize(x):
            beta, lambda_ = x
            res = pseudo_bayes_matching_run_se(alpha, sigma, beta, lambda_, use_logit_data=use_logistic_data, tolerance=se_tolerance)
            m, q, v = res['m'], res['q'], res['V']
            loss = generalisation_loss_logit_teacher(1.0, m, q, v, sigma)
            print(f'Trying for beta, lambda = {beta, lambda_}, loss is {loss}')
            return loss
    else:
        _, kappa1, kappastar = utility.KERNEL_COEFICIENTS[student_activation]
        add_var = utility.get_additional_noise_from_kappas(kappa1, kappastar, gamma)
        def to_minimize(x):
            beta, lambda_ = x
            # NOTE : old version of the code
            # res = pseudo_bayes_random_features_run_se(alpha, sigma, beta, lambda_, kappa1, kappastar, gamma, tolerance=se_tolerance, use_logit_data=use_logistic_data)
            # m, q, V = res['m'], res['q'], res['V']
            m, q, v = gcmpyo3.state_evolution.pseudo_bayes_state_evolution(alpha, beta, sigma**2 + add_var, gamma, kappa1, kappastar, lambda_, 1.0 - add_var, "probit", se_tolerance, True)
            # likewise, here rho is not 1.0 but we can probably use it nonetheless
            rho = 1.0 - utility.get_additional_noise_from_kappas(kappa1, kappastar, gamma)
            loss = generalisation_loss_logit_teacher(1.0, m, q, v, sigma)
            return loss

    if not optimize_beta:
        opt_res = optimize.minimize_scalar(lambda lambda_ : to_minimize([1.0, lambda_]),
                                        method = 'bounded',
                                        bounds =[lambda_min, lambda_max])
        lambda_opt = opt_res.x
        return 1.0, lambda_opt
    if optimize_beta:
        opt_res = optimize.minimize(to_minimize, x0 = [1.0, 0.5 * (lambda_max + lambda_min)], bounds =[(beta_min, beta_max), (lambda_min, lambda_max)], method='bounded')
        beta_opt, lambda_opt = opt_res.x
        return beta_opt, lambda_opt

##### FONCTIONS POUR ERM #####

def compute_optimal_lambda_for_test_error(alpha, sigma, gamma, student_activation, lambda_min, lambda_max, se_tolerance, matching, use_logistic_data, rho):    
    data_model = ["probit", "logit"][use_logistic_data]
    if matching:
        def to_minimize(lambda_):
            m, q, v, mhat, qhat, vhat = gcmpyo3.state_evolution.erm_state_evolution_matching(alpha, sigma**2, lambda_, rho, data_model, se_tolerance, False, False)
            # Minimizing the test error amounts to minimizing the quantity -m / np.sqrt(q) (should still hold for logistic data )
            return -m / np.sqrt(q)
    else:
        _, kappa1, kappastar = utility.KERNEL_COEFICIENTS[student_activation]
        
        def to_minimize(lambda_):
            print('Trying for lambda = ', lambda_)
            # NOTE : old version
            # res  = gcm_logistic_kappa_run_se(kappastar, kappa1, sigma, alpha, gamma, lambda_, tolerance=se_tolerance, stop_threshold=float('inf'), use_logistic_data=use_logistic_data)
            # m, q = res['m'], res['q']
            m, q, v, mhat, qhat, vhat = gcmpyo3.state_evolution.erm_state_evolution_gcm(alpha, sigma**2, gamma, kappa1, kappastar, lambda_, rho, data_model, se_tolerance, False)
            # Minimizing the test error amounts to minimizing this
            return - m / np.sqrt(q)

    opt_res = optimize.minimize_scalar(lambda lambda_ : to_minimize(lambda_),
                                       method = 'bounded',
                                       bounds =[lambda_min, lambda_max],
                                       options={'xatol' : DEFAULT_XATOL})
    lambda_opt = opt_res.x
    return lambda_opt

def compute_optimal_lambda_for_test_loss(alpha, sigma, gamma, student_activation, lambda_min, lambda_max, se_tolerance, matching, use_logistic_data : bool, rho, laplace_approximation : bool = False):
    data_model = ['probit', 'logit'][int(use_logistic_data)]
    # signature : generalisation_loss_probit_teacher(rho, m, q, student_variance, teacher_variance)
    loss_function = generalisation_loss_logit_teacher if data_model == 'logit' else generalisation_loss_probit_teacher

    if matching:
        def to_minimize(lambda_):
            m, q, v, mhat, qhat, vhat = gcmpyo3.state_evolution.erm_state_evolution_matching(alpha, sigma**2, lambda_, rho, data_model, se_tolerance, False, False)

            if laplace_approximation:
                local_field_variance = omega_inv_hessian_trace_matching(lambda_, vhat)
                return loss_function(rho, m, q, local_field_variance, sigma)
            else:
                return loss_function(rho, m, q, 0.0, sigma**2)

    else:
        _, kappa1, kappastar = utility.KERNEL_COEFICIENTS[student_activation]
        add_var = utility.get_additional_noise_from_kappas(kappa1, kappastar, gamma)
        def to_minimize(lambda_):
            m, q, v, mhat, qhat, vhat = gcmpyo3.state_evolution.erm_state_evolution_gcm(alpha, sigma**2, gamma, kappa1, kappastar, lambda_, rho, "logit", se_tolerance, False)
            # res          = gcm_logistic_kappa_run_se(kappastar, kappa1, sigma, alpha, gamma, lambda_, tolerance=se_tolerance, stop_threshold=float('inf'), use_logistic_data=use_logistic_data)
            # m, q, Vhat   = res['m'], res['q'], res['Vhat']
            # Minimizing the test error amounts to minimizing this

            if laplace_approximation:
                local_field_variance = omega_inv_hessian_trace_random_features(kappa1, kappastar, gamma, lambda_, vhat)
                return loss_function(rho, m, q, local_field_variance, sigma)
            else:
                return loss_function(rho, m, q, 0.0, sigma**2)

    opt_res = optimize.minimize_scalar(lambda lambda_ : to_minimize(lambda_),
                                       method = 'bounded',
                                       bounds =[lambda_min, lambda_max],
                                       options={'xatol' : DEFAULT_XATOL})
    lambda_opt = opt_res.x
    return lambda_opt

##############################

def save_config_and_result(filename, alpha, gamma, sigma, activation, minimized_quantity, lambda_opt, beta_opt, data_model):
    # header = ['alpha', 'gamma', 'sigma', 'activation', 'minimized_quantity', 'lambda_opt', 'beta_opt', 'data_model'] ou sans le beta_opt selon le fichier
    if not beta_opt is None:
        row = list(map(str, [alpha, gamma, sigma, activation, minimized_quantity, lambda_opt, beta_opt, data_model]))
    else:
        row = list(map(str, [alpha, gamma, sigma, activation, minimized_quantity, lambda_opt, data_model]))
    with open(filename, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--alpha', type=float, required=True, nargs='+')
    parser.add_argument('--sigma', type=float, required=True)
    parser.add_argument('--activation', type=str, default='erf')
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--lambda_min', type=float, default=DEFAULT_LAMBDA_MIN)
    parser.add_argument('--lambda_max', type=float, default=DEFAULT_LAMBDA_MAX)
    parser.add_argument('--se_tolerance', type=float, default=DEFAULT_SE_TOLERANCE)
    # if False, will use the probit data model
    parser.add_argument('--use_logistic_data', action='store_true', default=False)

    group0 = parser.add_mutually_exclusive_group(required = False)
    group0.add_argument('--laplace_approximation', action='store_true', default=False)
    group0.add_argument('--pseudo_bayes', action='store_true', default=False)

    parser.add_argument('--use_bayesian_opt', action='store_true', default=False)

    group1 = parser.add_mutually_exclusive_group(required = True)
    group1.add_argument('--gamma', type=float)
    # When we want to plot against 1 / alpha with the ratio n / p fixed so gamma changes
    group1.add_argument('--ratioNP', type=float)
    group1.add_argument('--matching', action='store_true')

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--loss', action='store_true')
    group2.add_argument('--error', action='store_true')
    group2.add_argument('--evidence', action='store_true')
    # only available for pseudo_bayes ! 

    parser.add_argument('--optimize_beta', default=False, action='store_true')

    args = parser.parse_args()

    # ugly code 
    USE_BAYESIAN_OPT = args.use_bayesian_opt
        
    for alpha in args.alpha:
        beta_opt = None
        debut = time()
        # compute gamma is the ratio n / p was given
        if args.matching: 
            gamma    = 1.0
            matching = True
            activation = 'matching'
        else:    
            matching = False
            activation = args.activation
            if args.ratioNP:
                gamma = args.ratioNP / alpha
            else:
                gamma = args.gamma
            
        if args.verbose:
            print(f'Starting for alpha = {alpha}, gamma = {gamma}')

        if args.loss:
            minimized_quantity = 'loss'
            if args.laplace_approximation:
                minimized_quantity = 'loss_laplace'
            if args.pseudo_bayes:
                minimized_quantity = 'loss_pb'
                beta_opt, lambda_opt = pseudo_bayesian_compute_optimal_lambda_for_loss(alpha, args.sigma, gamma, activation, args.lambda_min, args.lambda_max, args.se_tolerance, matching, args.use_logistic_data, args.optimize_beta, DEFAULT_BETA_MIN, DEFAULT_BETA_MAX)
            else:
                lambda_opt = compute_optimal_lambda_for_test_loss(alpha, args.sigma, gamma, activation, args.lambda_min, args.lambda_max, args.se_tolerance, matching, args.use_logistic_data, args.rho, args.laplace_approximation)
        elif args.error:
            if args.laplace_approximation:
                raise Exception('Useless to optimize for error with Laplace !')
            minimized_quantity = 'error'
            if args.pseudo_bayes:
                minimized_quantity = 'error_pb'
                beta_opt, lambda_opt = pseudo_bayesian_compute_optimal_lambda_for_error(alpha, args.sigma, gamma, activation, args.lambda_min, args.lambda_max, args.se_tolerance, matching, args.use_logistic_data, args.optimize_beta, DEFAULT_BETA_MIN, DEFAULT_BETA_MAX, args.rho)
            else:
                lambda_opt = compute_optimal_lambda_for_test_error(alpha, args.sigma, gamma, activation, args.lambda_min, args.lambda_max, args.se_tolerance, matching, args.use_logistic_data, args.rho)
        elif args.evidence:
            assert args.pseudo_bayes
            if not args.optimize_beta:
                minimized_quantity = 'evidence'
                lambda_opt = pseudo_bayesian_compute_optimal_lambda_for_evidence(alpha, args.sigma, gamma, activation, args.lambda_min, args.lambda_max, args.se_tolerance, matching, args.use_logistic_data, args.rho)
            else:
                minimized_quantity = 'evidence_beta_lambda'
                beta_opt, lambda_opt = pseudo_bayesian_compute_optimal_lambda_for_evidence(alpha, args.sigma, gamma, activation, args.lambda_min, args.lambda_max, args.se_tolerance, matching, args.use_logistic_data)
                print(f'Optimal beta, lambda are {beta_opt, lambda_opt}')
                # For now we don't save it but just print it 
                continue
        else:
            raise Exception('No quantity to minimize has been provided')

        print(f'Elapsed time for alpha = {alpha} : {time() - debut}')
        if not args.optimize_beta:
                beta_opt = None
                filename = DEFAULT_FILENAME
        else:
            filename = DEFAULT_FILENAME_LAMBDA_BETA
        data_model = "logit" if args.use_logistic_data else "probit"
        save_config_and_result(filename, alpha, gamma, args.sigma, activation, minimized_quantity, lambda_opt, beta_opt, data_model)