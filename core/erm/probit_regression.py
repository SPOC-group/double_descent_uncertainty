"""
Solve probit regression
NOTE : 
"""

CVXPY_IS_LOADED = False

try:
    import cvxpy as cv
    CVXPY_IS_LOADED = True
except:
    print("Could not import cvxpy, will not be able to use the function solve_probit_regression")

def solve_probit_regression(X, y, lambda_):
    if not CVXPY_IS_LOADED:
        raise Exception("cvxpy is not loaded !")
    n, d = X.shape
    
    w = cv.Variable(d)
    error = - cv.sum(cv.log_normcdf(cv.multiply(y, X @ w))) + (lambda_ / 2.0) * cv.sum_squares(w)
    obj = cv.Minimize(error)
    prob = cv.Problem(obj)
    prob.solve()

    weight = w.value
    return weight
