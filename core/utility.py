# utility.py
# various useful functions

import numpy as np
from scipy.integrate import quad
from scipy.optimize  import minimize_scalar, root_scalar
from scipy.special   import erfc
import scipy.stats   as stats

import gcmpyo3

# FUNCTION OF THE LOGIT MODEL 
sigmoid       = np.vectorize(lambda x : 1. / (1. + np.exp( -x )))
sigmoid_prime = np.vectorize(lambda x : sigmoid(x) * ( 1.0 - sigmoid(x) ) )

sigmoid_inv = np.vectorize(lambda y : np.log(y/(1-y)))

erf_prime = np.vectorize(lambda x : 2. / np.sqrt(np.pi) * np.exp(-x**2))
erfc_prime = np.vectorize(lambda x : -2. / np.sqrt(np.pi) * np.exp(-x**2))

bernoulli_variance = np.vectorize(lambda p : 4 * p * (1. - p))

def gaussian(x, mean=0, var=1):
    return np.exp(-.5 * (x-mean)**2/var) / np.sqrt(2*np.pi*var)

@np.vectorize
def probit(x):
    return stats.norm.cdf(x)
    # return 0.5 * erfc(- x / np.sqrt(2))

@np.vectorize
def probit_inv(p):
    return stats.norm.ppf(p)

def proximal_operator(func : callable, x : float, tau : float) -> float:
    to_minimize = lambda z : ((z - x)**2) / (2 * tau) + func(z)
    res = minimize_scalar(to_minimize, method='Golden')
    if res['x'] > 1e10:
        print(res['x'])
    return res['x']
    # res.x here is an array with a single element inside

# === DAMPING USED E.G. to compute 

def damping(q_new : float, q_old : float, coef_damping : float =0.5) -> float:
    if q_old == float('inf') or np.isnan(q_old):
        return q_new
    return (1 - coef_damping) * q_new + coef_damping * q_old

# === coefficients for the Random Features kernel

# in the order : kappa0, kappa1, kappastar
KERNEL_COEFICIENTS = {'relu': (1/np.sqrt(2*np.pi), 0.5, np.sqrt((np.pi-2)/(4*np.pi))), 
                      'erf': (0, 2/np.sqrt(3*np.pi), 0.200364),
                      'tanh': (0, 0.605706, 0.165576),
                      'sign': (0, np.sqrt(2/np.pi), np.sqrt(1-2/np.pi)),
                      'matching': (None, None, None)
                      }

def build_gcm_from_activation(student_activation, gamma, p = 256):
    if student_activation == 'matching':
        assert gamma == 1.0, 'Gamma must be one for identity feature'
        return np.eye(p), np.eye(p), np.eye(p)
        
    d = int(gamma * p)
    try:
        _, kappa1, kappastar = KERNEL_COEFICIENTS[student_activation]
    except Exception as e:
        print('Activation not known')
        raise e
    Psi   = np.eye(p)
    
    F_student = np.random.normal(0,1, (d, p)) / np.sqrt(p) # student random projection
    Omega = kappa1**2 * F_student @ F_student.T + kappastar**2 * np.identity(d)
    Phi   = kappa1 * F_student.T
    return Psi, Omega, Phi

def get_change_matrices_from_activation(activation, gamma, p = 256):
    try:
        _, kappa1, kappastar = KERNEL_COEFICIENTS[activation]
    except Exception as e:
        print('Activation not known')
        raise e
    Psi   = np.eye(p)
    d = int(gamma * p)
    
    F_student = np.random.normal(0,1, (d, p)) / np.sqrt(p) # student random projection
    return kappa1 * F_student, kappastar 

def get_additional_noise_from_kappas(kappa1, kappastar, gamma):
    """
    Returns the variance 
    """
    kk1, kkstar               = kappa1**2, kappastar**2
    lambda_minus, lambda_plus = (1. - np.sqrt(gamma))**2, (1. + np.sqrt(gamma))**2
    
    def to_integrate(lambda_, kk1, kkstar, lambda_minus, lambda_plus):
        return np.sqrt((lambda_plus - lambda_) * (lambda_ - lambda_minus)) / (kkstar + kk1 * lambda_)
    
    return 1.0 - kk1 * quad(lambda lambda_ : to_integrate(lambda_, kk1, kkstar, lambda_minus, lambda_plus), lambda_minus, lambda_plus)[0] / (2 * np.pi)

def get_additional_noise_from_activation(activation, gamma):
    _, kappa1, kappastar = KERNEL_COEFICIENTS[activation]
    return get_additional_noise_from_kappas(kappa1, kappastar, gamma)

# === Laplace approximation prediction 

# the functions to get the variance of Laplace

def omega_inv_hessian_trace_random_features(kappa1, kappastar, gamma, lambda_, Vhat):
    """
    Gives the asymptotic value of x . H^{-1} . x -> Tr(H^{-1} Omega) where the covariance of x is Omega
    """
    kk1 = kappa1**2
    kkstar = kappastar**2
    return mp_integral(lambda x : (kk1 * x + kkstar) / (lambda_ + Vhat * (kk1 * x + kkstar)), gamma)

def omega_inv_hessian_trace_matching(lambda_, Vhat):
    return 1. / (lambda_ + Vhat)

LOGISTIC_APPROX_COEF = 0.5875651988237005

def get_rescaled_m_q(m, q, V, sigmoid_to_probit_scaling = LOGISTIC_APPROX_COEF ):
    """
    Laplace here is equivalent to rescaling m and q. Here, we compute m_laplace and q_laplace
    Probably works also with pseudo-Bayes but not sure (TODO : Check that it works)
    arguments: 
        - eta := sigmoid_to_probit_scaling is such that sigmoid(x) = probit(eta * x)
        As in Kristiadi et al. this value is sqrt(pi / 8) but we can fine tune it
    returns: 
        - m, q
    """ 
    scaling = np.sqrt(1.0 + sigmoid_to_probit_scaling**2 * np.array(V))
    return m / scaling, q / scaling**2

def sigmoid_approximated_by_probit(x):
    """
    Approximates logistic function with the probit
    """
    return probit(LOGISTIC_APPROX_COEF * x)

def inverse_of_sigmoid_approximated_by_probit(y):
    return (1. / LOGISTIC_APPROX_COEF) * stats.norm.ppf(y)

def averaged_sigmoid(x, V):
    """
    Prediction with the logistic model given that the local field is Gaussian with mean x and variance V. Useful for laplace 
    where the variance will be x^t H x, but also for the pseudo-Bayesian learning (cf. Moulines) where the V is given by state-evolution (I think, to check
    w/ experiments)
    """
    mean = x
    std  = np.sqrt(V)
    
    return quad(lambda z : sigmoid(z * std + mean) * stats.norm.pdf(z, loc=0.0, scale=1.0), -10.0, 10.0)[0]

def averaged_sigmoid_prime(x, variance):
    if variance > 0.0:
        model = gcmpyo3.NormalizedPseudoBayesLogistic(beta = 1.0)
        return model.call_dz0(1.0, x, variance)
    else:
        return sigmoid(x) * (1.0 - sigmoid(x))

def approximated_averaged_sigmoid(x, V):
    # divide by sqrt(8 / pi) <=> multiply by sqrt(pi / 8)
    # Is equal to probit(x / np.sqrt(1 + V * LOGISTIC_APPROX_COEF**2))
    return stats.norm.cdf(x / np.sqrt(1. / LOGISTIC_APPROX_COEF**2 + V))

def approximated_average_sigmoid_inv(p, V):
    """
    Sert pour le pseudo bayesien et pour Laplace
    """
    return np.sqrt( (1. / LOGISTIC_APPROX_COEF) + V) * stats.norm.ppf(p)

def exact_average_logistic_inverse(p, V):
    def likelihood(z):
        return quad(lambda x : sigmoid(z + x * np.sqrt(V)) * stats.norm.pdf(x, loc = 0.0, scale = 1.0), -20.0, 20.0)[0]

    def likelihood_prime(z):
        return quad(lambda x : sigmoid_prime(z * x * np.sqrt(V)) * stats.norm.pdf(x, loc = 0.0, scale = 1.0), -20.0, 20.0)[0]

    sol = root_scalar(lambda z : likelihood(z) - p, x0 = approximated_average_sigmoid_inv(p, V), fprime = likelihood_prime)
    return sol.root

## 

def generalisation_error_probit_teacher(rho, m, q, sigma):
    return 1. / np.pi * np.arccos(m / (np.sqrt(q * (rho + sigma**2))))

def generalisation_error_logit_teacher(rho, m, q, sigma = 0.0):
    model = gcmpyo3.Logit ( noise_variance = 0.0 )

    def integrand_plus(xi):
        # Probability that we make a mistake when the xabel y = -1 (and not 1 !) because it's the proba that 
        # the teacher has the label 1 given that the student has a -1 i.e a local field xi < 0
        # so if xi > 0, there is not error
        if xi > 0.0:
            return 0.0
        # Caution ! Needs to be normalized !!! 
        # If xi < 0 i.e predicted label is -1, we return the proba tht the true label is 1 
        return model.call_z0(1.0, m / np.sqrt(q) * xi, rho - m * m / q + sigma**2)

    def integrand_minus(xi):
        if xi < 0.0:
            return 0.0
        # Caution ! Needs to be normalized !!! 
        return model.call_z0(-1.0, m / np.sqrt(q) * xi, rho - m * m / q + sigma**2)
        # return model.call_z0(-1, m / np.sqrt(q) * xi, rho - m**2 / q)
    
    I_plus  = quad(lambda xi : integrand_plus(xi) * stats.norm.pdf(xi, loc = 0.0, scale = 1.0), -10.0, 10.0)[0]
    I_minus = quad(lambda xi : integrand_minus(xi) * stats.norm.pdf(xi, loc = 0.0, scale = 1.0), -10.0, 10.0)[0]

    return I_minus + I_plus


def generalisation_loss_logit_teacher(rho, m, q, student_variance, teacher_variance):
    bound = 10.0
    teacher    = gcmpyo3.Logit(noise_variance = 0.0)

    if student_variance != 0.0:
        # NOTE : Make sure that the Z0 is normalized ! 
        # LogisticDataModel because we average over logistic activation
        student = gcmpyo3.Logit(noise_variance = 0.0)
        student_Z0 = lambda y, omega : - np.log( student.call_z0(y, omega, student_variance))
    if student_variance == 0.0:
        student_Z0 = lambda y, omega : np.log(1 + np.exp(- y * omega))
    
    loss = 0.0
    for y in [-1.0, 1.0]:
        loss += quad(lambda xi : teacher.call_z0(y, m / np.sqrt(q) * xi, rho - m**2 / q + teacher_variance) * student_Z0(y, np.sqrt(q) * xi) * stats.norm.pdf(xi, loc = 0.0, scale = 1.0), 
                               -bound, bound)[0]
    return loss

def generalisation_loss_probit_teacher(rho, m, q, student_variance, teacher_variance):
    """
        It's still going to be with a logistic student
    """
    bound = 10.0
    teacher    = gcmpyo3.Probit(noise_variance = 0.0)

    if student_variance != 0.0:
        # NOTE : Make sure that the Z0 is normalized ! 
        # LogisticDataModel because we average over logistic activation
        student = gcmpyo3.Logit(noise_variance = 0.0)
        student_Z0 = lambda y, omega : - np.log( student.call_z0(y, omega, student_variance))
    if student_variance == 0.0:
        student_Z0 = lambda y, omega : np.log(1 + np.exp(- y * omega))
    
    loss = 0.0
    for y in [-1.0, 1.0]:
        loss += quad(lambda xi : teacher.call_z0(y, m / np.sqrt(q) * xi, rho - m**2 / q + teacher_variance) * student_Z0(y, np.sqrt(q) * xi) * stats.norm.pdf(xi, loc = 0.0, scale = 1.0), 
                               -bound, bound)[0]
    return loss

## 

def calibration_logit_teacher(level, m, q, rho, student_noise_var, teacher_noise_var):
    local_field = gcmpyo3.utility.exact_inverse_averaged_sigmoid(level, student_noise_var)
    
    return level - gcmpyo3.utility.conditional_expectation_logit(m, q, teacher_noise_var , rho, local_field)

def calibration_probit_teacher(level, m, q, rho, student_noise_var, teacher_noise_var):
    local_field = gcmpyo3.utility.exact_inverse_averaged_sigmoid(level, student_noise_var)
    
    # closed form expression
    return level - probit( m / q * local_field / np.sqrt(rho - m**2 / q + teacher_noise_var) )

def average_positive_confidence(q, v):
    # taking rho = m = q = 1.0 should give mean = z * np.sqrt(q) and variance = v 
    return quad(lambda z : gcmpyo3.utility.conditional_expectation_logit(1.0, 1.0, v, 1.0, z * np.sqrt(q)) * stats.norm.pdf(z, loc = 0.0, scale = 1.0), 0.0, 20.0)[0]

## 

def ece_logit_teacher(m, q,rho, student_noise_var, teacher_noise_var, upper_bound = 0.99999):
    def to_integrate(level):
        lf = gcmpyo3.utility.exact_inverse_averaged_sigmoid(level, student_noise_var)
        return np.abs(level - gcmpyo3.utility.conditional_expectation_logit(m, q, teacher_noise_var, rho, lf)) * \
                        stats.norm.pdf(lf, loc = 0.0, scale = np.sqrt(q)) / np.abs(averaged_sigmoid_prime(lf, student_noise_var))
    return quad(to_integrate, 0.5, upper_bound)[0]

def ece_probit_teacher(m, q, rho, student_noise_var, teacher_noise_var, upper_bound = 0.99999):
    def to_integrate(level):
        lf = gcmpyo3.utility.exact_inverse_averaged_sigmoid(level, student_noise_var)
        return np.abs(level - probit( m / q * lf / np.sqrt(rho - m**2 / q + teacher_noise_var) )) * \
                        stats.norm.pdf(lf, loc = 0.0, scale = np.sqrt(q)) / np.abs(averaged_sigmoid_prime(lf, student_noise_var))
    return quad(to_integrate, 0.5, upper_bound)[0]

## 

def conditional_variance_logit_teacher(level, m, q, rho, student_noise_var, teacher_noise_var):
    lf = gcmpyo3.utility.exact_inverse_averaged_sigmoid(level, student_noise_var)
    return gcmpyo3.utility.conditional_variance_logit(m, q, teacher_noise_var, rho, lf)

def conditional_variance_probit_teacher(level, m, q, rho, student_noise_var, teacher_noise_var):
    lf = gcmpyo3.utility.exact_inverse_averaged_sigmoid(level, student_noise_var)
    return gcmpyo3.utility.conditional_variance_probit(m, q, teacher_noise_var, rho, lf)

### === Marcenko-Pastur 

def mp_integral(f : callable, gamma):
    """
    integrates an arbitrary function against MP distriubtion
    """
    lambda_minus, lambda_plus = (1. - np.sqrt(gamma))**2, (1. + np.sqrt(gamma))**2
    integral = quad(lambda x : f(x) * np.sqrt((lambda_plus - x) * (x - lambda_minus)) / (2 * np.pi * gamma * x), lambda_minus, lambda_plus)[0]
    if gamma > 1.0:
        return integral + (1.0 - 1.0 / gamma) * f(0.0)
    return integral

def mp_integral_without_zero(f : callable, gamma):
    """
    integrates an arbitrary function against MP distriubtion
    """
    lambda_minus, lambda_plus = (1. - np.sqrt(gamma))**2, (1. + np.sqrt(gamma))**2
    integral = quad(lambda x : f(x) * np.sqrt((lambda_plus - x) * (x - lambda_minus)) / (2 * np.pi * gamma * x), lambda_minus, lambda_plus)[0]
    return integral

### Density

def get_teacher_student_density(m, q, v, delta, rho, N = 100):
    cov_matrix = np.array([
        [rho, m],
        [m, q]
    ])

    density = np.zeros((N, N))
    p_range = np.linspace(1e-4, 1.0 - 1e-4, N)
    for i in range(N):
        for j in range(N):
            p_student = p_range[j]
            p_teacher = p_range[i]
            if v > 0.0:
                inverse_student = gcmpyo3.utility.exact_inverse_averaged_sigmoid(p_student, v)
                derv_student    = averaged_sigmoid_prime(inverse_student, v)
            else:
                inverse_student = sigmoid_inv(p_student)
                derv_student    = p_student * (1.0 - p_student)
            # Take the additional mismatch noise into account
            inverse_teacher = gcmpyo3.utility.exact_inverse_averaged_sigmoid(p_teacher, delta)
            derv_teacher    = averaged_sigmoid_prime(inverse_teacher, delta)
            density[i][j] = stats.multivariate_normal.pdf([inverse_teacher, inverse_student], mean = [0.0, 0.0], cov = cov_matrix) / (derv_teacher * derv_student)
    return p_range, density

def get_teacher_student_density_probit_teacher(m, q, v, delta, rho, N = 100):

    cov_matrix = np.array([
        [rho, m],
        [m, q]
    ])

    density = np.zeros((N, N))
    p_range = np.linspace(1e-6, 1.0 - 1e-6, N)
    for i in range(N):
        for j in range(N):
            p_student = p_range[j]
            p_teacher = p_range[i]
            if v > 0.0:
                inverse_student = gcmpyo3.utility.exact_inverse_averaged_sigmoid(p_student, v)
                derv_student    = averaged_sigmoid_prime(inverse_student, v)
            else:
                inverse_student = sigmoid_inv(p_student)
                derv_student    = p_student * (1.0 - p_student)
            # Take the additional mismatch noise into account
            inverse_teacher   = probit_inv(p_teacher) * np.sqrt(delta)
            derv_teacher      = stats.norm.pdf(inverse_teacher / np.sqrt(delta), loc = 0.0, scale = 1.0) / np.sqrt(delta)
            density[i][j] = stats.multivariate_normal.pdf([inverse_teacher, inverse_student], mean = [0.0, 0.0], cov = cov_matrix) / np.abs(derv_teacher * derv_student)
    return p_range, density

# TODO : Write the training error 

def logistic_loss(x):
    return np.log(1.0 + np.exp(-x))

def moreau_loss(x, y, omega, v):
    return (x-omega)**2/(2*v) + logistic_loss(y*x)

def integrand_training_loss_plus_logit_teacher(xi, M, Q, V, Vstar):
    model = gcmpyo3.Logit ( noise_variance = 0.0 )

    omega = np.sqrt(Q)*xi
    omegastar = (M/np.sqrt(Q))*xi
#   lambda_star_plus = np.float(mpmath.findroot(lambda lambda_star_plus: lambda_star_plus - omega - V/(1 + np.exp(np.float(lambda_star_plus))), 10e-10))
    lambda_star_plus = minimize_scalar(lambda x: moreau_loss(x, 1, omega, V))['x']
    
    # l_plus = logistic_loss(lambda_star_plus)
    if lambda_star_plus > 0.0:
        return 0.0

    # return model.call_z0(1.0, omegastar, Vstar) * l_plus
    return model.call_z0(1.0, omegastar, Vstar)


def integrand_training_loss_minus_logit_teacher(xi, M, Q, V, Vstar):
    model = gcmpyo3.Logit ( noise_variance = 0.0 )
    omega = np.sqrt(Q)*xi
    omegastar = (M/np.sqrt(Q))*xi
#   lambda_star_minus = np.float(mpmath.findroot(lambda lambda_star_minus: x - omega + V/(1 + np.exp(-np.float(lambda_star_minus))), 10e-10))
    lambda_star_minus = minimize_scalar(lambda x: moreau_loss(x, -1, omega, V))['x']
    
    # l_minus = logistic_loss(-lambda_star_minus)
    if lambda_star_minus < 0.0:
        return 0.0

    # return model.call_z0(-1.0, omegastar, Vstar) * l_minus
    return model.call_z0(-1.0, omegastar, Vstar)

def training_error_logit_teacher(m, q, v, vstar):
    # Returns the training ERROR (and not the loss ...) for ERM
    I1 = quad(lambda xi: integrand_training_loss_plus_logit_teacher(xi, m, q, v, vstar) * stats.norm.pdf(xi), -10, 10, limit=500)[0]
    I2 = quad(lambda xi: integrand_training_loss_minus_logit_teacher(xi, m, q, v, vstar) * stats.norm.pdf(xi), -10, 10, limit=500)[0]
    return I1 + I2