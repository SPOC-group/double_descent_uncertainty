import numpy as np
import sklearn.linear_model as linear_model

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

