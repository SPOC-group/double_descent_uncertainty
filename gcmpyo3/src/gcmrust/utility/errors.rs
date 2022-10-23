// errors.rs
// script to compute the error and loss of estimators

use std::f64::consts::PI;

pub fn error_probit_model(m : f64, q : f64, rho : f64, delta : f64) -> f64 {
    return (m / (q * (rho + delta)).sqrt()).acos() / PI;
}
