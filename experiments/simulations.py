# simulations.py
# Contains simulations to sample the estimators with Gaussian data

import sys
sys.path.append('..')

import numpy as np
import scipy.linalg
from tqdm import tqdm 

import core.utility as utility
import core.amp.likelihood.bo_logit_likelihood as bo_logit_likelihood
import core.amp.likelihood.logit_likelihood as logit_likelihood
import core.amp.prior.nonseparable_gaussian_prior as nonseparable_gaussian_prior
import core.erm.logistic_regression as logistic_regression
import core.amp as amp

def generate_logit_data(kappa1, kappastar, F, n, teacher_dim, student_dim, noise_std, wstar = None):
    if wstar is None:
        wstar = np.random.normal(0.0, 1.0, size=teacher_dim)
        wstar = wstar * np.sqrt(teacher_dim) / np.linalg.norm(wstar)

    X0 = np.random.normal(0.0, 1.0, size=(n, teacher_dim))
    V  = kappa1 * X0 @ F.T  + kappastar * np.random.normal(0.0, 1.0, size=(n, student_dim))
    Y = 2.0 * np.random.binomial(1.0, p = utility.sigmoid(X0 @ wstar / np.sqrt(teacher_dim) + noise_std * np.random.normal(0.0, 1.0, size=n)) ) - 1.0
    return wstar, V / np.sqrt(student_dim), Y

def inv_hessian(w, X, Y, lambda_):
    p = len(w)
    D = np.diag([utility.sigmoid(w @ x) * (1.0 - utility.sigmoid(w @ x)) for x in X])
    hessian     = X.T @ D @ X + lambda_ * np.eye(p)
    return np.linalg.inv(hessian)

def inv_hessian_trace(w, X, Y, lambda_, Omega):
    p = len(w)
    return np.trace(inv_hessian(w, X, Y, lambda_) @ Omega / p)

def erm_trial(d, n_over_d, inv_alpha_range, lambda_list, kappa1, kappastar, noise_std, F_global):
    """
    arguments:
        - d : dimension of the teacher
        - n_over_d : ratio #sample / d
        - inv_alpha_range : list of (p / n) where p = dimension of the student
        - lambda_list : L2 reg. strength
        - kappa1, kappastar : moments of the activation
        - noise_std : standard deviation of the noise in the teacher 
        - F_global : P x d matrix represneting the random features, where P = max(student dimension in inv_alpha_range)
    returns :
        - m_list
        - q_list
        - hessian_list : list of Tr(H Omega)
    """
    hessian_list, m_list, q_list = [], [], [] 

    for inv_alpha, lambda_ in tqdm(zip(inv_alpha_range, lambda_list)):
        gamma = n_over_d * inv_alpha
        n = int(n_over_d * d)
        
        # student_dim 
        p     = int(gamma * d)
        
        # build the matrices / covariance matrices
        F = F_global[:p]
        wstar, X, Y = generate_logit_data(kappa1, kappastar, F, n, d, p, noise_std)
        
        what     = logistic_regression.solve_logistic_regression(X, Y, lambda_)

        Omega    = kappa1**2 * F @ F.T + kappastar**2 * np.eye(p)
        m_list.append(kappa1 * wstar @ F.T @ what / np.sqrt(p * d))
        q_list.append(what @ Omega @ what / p)
        hessian_list.append( inv_hessian_trace(what, X, Y, lambda_, Omega) )

    return m_list, q_list, hessian_list

def bo_trial(d, n_over_d, inv_alpha_range, kappa1, kappastar, noise_std, F_global):
    """
    returns:
        - projection of wstar on the subset spanned by 
        - estimator what and vhat
    """
    m_list, q_list, v_list = [], [], []
    n = int(n_over_d * d)

    wstar = np.random.multivariate_normal(np.zeros(d), np.eye(d))
    wstar = wstar / np.linalg.norm(wstar) * np.sqrt(d)

    for inv_alpha in tqdm(inv_alpha_range):
        gamma = n_over_d * inv_alpha
        # student_dim 
        p     = int(gamma * d)

        # build the matrices / covariance matrices
        F = F_global[:p]
        Phi   = kappa1 * F.T
        
        Omega              = kappa1**2 * F @ F.T + kappastar**2 * np.eye(p)
        Omega_inv          = np.linalg.inv(Omega)
        Omega_inv_sqrt     = np.real(scipy.linalg.sqrtm(Omega_inv))
        new_Phi            = Phi @ Omega_inv_sqrt
        teacher_covariance = new_Phi.T @ new_Phi

        additional_noise   = utility.get_additional_noise_from_kappas(kappa1, kappastar, gamma)
        new_noise_var      = additional_noise + noise_std**2
        new_noise_std      = np.sqrt(new_noise_var)

        X = np.random.normal(0.0, 1.0, (n, p))
        Y = 2.0 * np.random.binomial(1.0, p = utility.sigmoid(X @ new_Phi.T @ wstar / np.sqrt(d) + new_noise_std * np.random.normal(0.0, 1.0, size=n)) ) - 1.0
        X = X / np.sqrt(d)

        # run amp
        likelihood         =  bo_logit_likelihood.BOLogitLikelihood( noise_var = new_noise_var )
        prior              = nonseparable_gaussian_prior.NonSepGaussianPrior(teacher_covariance)

        retour = amp.gamp.iterate_gamp(X, Y, wstar, likelihood, prior, verbose = False, tol = 1e-3)
        what, vhat = retour['estimator'], retour['variances']

        projected_wstar = wstar @ new_Phi

        m_list.append( projected_wstar @ what / d)
        q_list.append( what @ what / d )
        v_list.append( np.sum(vhat) / d )
        
    return m_list, q_list, v_list

def eb_trial(d, n_over_d, inv_alpha_range, lambda_list, kappa1, kappastar, noise_std, F_global):
    n = int(n_over_d * d)
    m_list, q_list, v_list = [], [], []

    wstar = np.random.multivariate_normal(np.zeros(d), np.eye(d))
    wstar = wstar / np.linalg.norm(wstar) * np.sqrt(d)

    for inv_alpha, lambda_ in tqdm(zip(inv_alpha_range, lambda_list)):
        gamma = n_over_d * inv_alpha
        p     = int(gamma * d)
                
        F     = F_global[:p]
        Phi   = kappa1 * F.T
        
        Omega              = kappa1**2 * F @ F.T + kappastar**2 * np.eye(p)
        Omega_inv          = np.linalg.inv(Omega)
        Omega_inv_sqrt     = np.real(scipy.linalg.sqrtm(Omega_inv))
        new_Phi            = Phi @ Omega_inv_sqrt

        new_noise_var      = noise_std**2 + np.trace(np.eye(d) - new_Phi @ new_Phi.T) / d
        new_noise_std      = np.sqrt(new_noise_var)

        student_X = np.random.normal(0.0, 1.0, (n, p))
        Y = 2.0 * np.random.binomial(1.0, p = utility.sigmoid(student_X @ new_Phi.T @ wstar / np.sqrt(d) + new_noise_std * np.random.normal(0.0, 1.0, size=n)) ) - 1.0

        student_X = student_X / np.sqrt(p)

        cov = Omega / (lambda_)
        prior              = nonseparable_gaussian_prior.NonSepGaussianPrior(cov)

        likelihood         = logit_likelihood.LogitLikelihood(beta = 1.0, normalized = True)
        
        retour = amp.gamp.iterate_gamp(student_X, Y, wstar, likelihood, prior, verbose = False, tol = 1e-3)
        what = retour['estimator']
        vhat = retour['variances']
        
        m_list.append( (wstar @ new_Phi @ what) / np.sqrt(p * d) )
        q_list.append((what @ what) / p)
        v_list.append( np.sum(vhat) / p)

    return m_list, q_list, v_list