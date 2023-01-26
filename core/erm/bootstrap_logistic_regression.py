# script to run bootstrap on logistic classification 

import numpy as np
import sklearn.linear_model as linear_model

sigmoid = lambda x : 1.0 / (1.0 + np.exp(-x))

def solve_logistic_regression(X, y, lambda_):
    """
    Returns the estimator 
    """
    max_iter = 10000
    tol      = 1e-16

    if lambda_ > 0.:
        lr = linear_model.LogisticRegression(penalty='l2',solver='lbfgs',fit_intercept=False, 
                                  C = (1. / lambda_), max_iter=max_iter, tol=tol, verbose=0)
    else:
        lr = linear_model.LogisticRegression(penalty='none', solver='lbfgs', fit_intercept=False, max_iter=max_iter, tol=tol, verbose=0)
    lr.fit(X, y)

    if lr.n_iter_ == max_iter:
        print('Attention : logistic regression reached max number of iterations ({:.2f})'.format(max_iter))

    w = lr.coef_[0]
    return w

def bootstrap_logistic_classification(X, Y, lambda_, n_resamples=1000):
    """
    NOTE : as n_resamples -> \infty, the average of the bootstrap weights should converge to the one of ERM
    The question is : what's the variance of the prediction for the confidence ? Does it contain the true one ? 
    """

    n, d = X.shape
    ws = np.zeros((n_resamples, d))

    for trial in range(n_resamples):
        indices = np.random.choice(d, size=n, replace=True)
        X_resample, Y_resample = X[indices], Y[indices]

        ws[trial] = solve_logistic_regression(X_resample, Y_resample, lambda_)
    
    return ws

def average_confidence_bootstrap(x, ws_bootstrap):
    # return the average of the confidence from each bootstrap
    return np.mean(sigmoid(ws_bootstrap @ x))

def variance_confidence_bootstrap(x, ws_bootstrap):
    return np.var(sigmoid(ws_bootstrap @ x))
