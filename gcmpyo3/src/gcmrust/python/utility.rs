use pyo3::prelude::*;

use crate::gcmrust::channels;
use crate::gcmrust::utility as ut;
use crate::gcmrust::data_models;
use crate::gcmrust::state_evolution as se;

#[pymodule]
pub fn utility(_py : Python, m : &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(exact_inverse_averaged_sigmoid, m)?)?;
    
    m.add_function(wrap_pyfunction!(conditional_expectation_logit, m)?)?;
    m.add_function(wrap_pyfunction!(conditional_expectation_probit, m)?)?;

    m.add_function(wrap_pyfunction!(conditional_variance_logit, m)?)?;
    m.add_function(wrap_pyfunction!(conditional_variance_probit, m)?)?;

    Ok(())
}


#[pyfunction]
fn exact_inverse_averaged_sigmoid(p : f64, variance : f64) -> f64 {
    return ut::approximation::exact_inverse_averaged_sigmoid(p, variance);
}

#[pyfunction]
fn conditional_expectation_logit(m : f64, q : f64, delta_teacher : f64, rho : f64, student_local_field : f64) -> f64 {
    return ut::approximation::conditional_expectation_logit(m, q, delta_teacher, rho, student_local_field);
}

#[pyfunction]
fn conditional_expectation_probit(m : f64, q : f64, delta_teacher : f64, rho : f64, student_local_field : f64) -> f64 {
    return ut::approximation::conditional_expectation_probit(m, q, delta_teacher, rho, student_local_field);
}

#[pyfunction]
fn conditional_variance_logit(m : f64, q : f64, delta_teacher : f64, rho : f64, student_local_field : f64) -> f64 {
    return ut::approximation::conditional_variance_logit(m, q, delta_teacher, rho, student_local_field);
}

#[pyfunction]
fn conditional_variance_probit(m : f64, q : f64, delta_teacher : f64, rho : f64, student_local_field : f64) -> f64 {
    return ut::approximation::conditional_variance_probit(m, q, delta_teacher, rho, student_local_field);
}