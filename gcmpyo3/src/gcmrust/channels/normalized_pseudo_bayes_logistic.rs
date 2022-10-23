use peroxide::numerical::*;
use pyo3::{pyclass, pymethods, PyResult};
use statrs::function::logistic;
use std::f64::consts::PI;

use crate::gcmrust::{data_models::base_partition::{Partition, NormalizedChannel}, utility::constants::*};

static NORMALIZED_PSEUDO_BAYES_BOUND : f64 = 10.0;

pub fn likelihood(x : f64, beta : f64) -> f64 {
    return logistic::logistic(beta * x);
}

pub fn z0_integrand(z : f64, y : f64, w : f64, sqrt_v : f64, beta : f64) -> f64 {
    let local_field : f64 = y * (z * sqrt_v + w);
    return likelihood(local_field, beta) * (- z*z / 2.0).exp() / (2.0 * PI).sqrt();
}

pub fn dz0_integrand(z : f64, y : f64, w : f64, sqrt_v : f64, beta : f64) -> f64 {
    let local_field : f64 = y * (z * sqrt_v + w);
    return z * likelihood(local_field, beta) * (- z*z / 2.0).exp() /  ((2.0 * PI).sqrt() * sqrt_v);
}

fn ddz0_integrand(z : f64, y : f64, w : f64, sqrt_v : f64, beta : f64) -> f64 { 
    return z.powi(2) * likelihood(y * (z * sqrt_v + w), beta) * (-z*z / 2.0).exp() / (2.0 * PI).sqrt();
}
        
#[pyclass(unsendable)]
pub struct NormalizedPseudoBayesLogistic {
    pub beta  : f64
}

impl NormalizedChannel for NormalizedPseudoBayesLogistic {
    
}

impl NormalizedPseudoBayesLogistic {
    fn integrate_function(&self, f : &dyn Fn(f64) -> f64) -> f64 {
        return integral::integrate(f, (-NORMALIZED_PSEUDO_BAYES_BOUND, NORMALIZED_PSEUDO_BAYES_BOUND), integral::Integral::G30K61(GK_PARAMETER));
        // return integral::integrate(f, (-NORMALIZED_PSEUDO_BAYES_BOUND, NORMALIZED_PSEUDO_BAYES_BOUND), integral::Integral::GaussLegendre(16));
    }

    pub fn unstable_derivative_z0_beta(&self, y : f64, w : f64, v : f64) -> f64 {
        return (1.0 / self.beta) * (self.ddz0(y, w, v) * v + self.z0(y, w, v) + w * self.dz0(y, w, v));
    }
}

impl Partition for NormalizedPseudoBayesLogistic{
    // TODO : Trouver un moyen de pas avoir a recopier le code ebntre les deux PB
    fn z0(&self, y : f64, w : f64, v : f64) -> f64 {
        // return z0(y, w, v, self.beta, INTEGRAL_BOUNDS);
        let sqrt_v = v.sqrt();
        return self.integrate_function(&|z : f64| -> f64 {z0_integrand(z, y, w, sqrt_v, self.beta)});
    }

    fn dz0(&self, y : f64, w : f64, v : f64) -> f64 {
        // return dz0(y, w, v, self.beta, INTEGRAL_BOUNDS);
        let sqrt_v = (v).sqrt();
        return self.integrate_function(&|z : f64| -> f64 {dz0_integrand(z, y, w, sqrt_v, self.beta)}) ;
    }

    fn ddz0(&self, y : f64, w : f64, v : f64) -> f64 {
        // return ddz0(y, w, v, self.beta, INTEGRAL_BOUNDS, None);
        let sqrt_v = v.sqrt();
        let z0 : f64 = self.z0(y, w, v);
    
        let integrale = self.integrate_function(&| z : f64| -> f64 {ddz0_integrand(z, y, w, sqrt_v, self.beta)});
        return - z0 / v + integrale / v;
    }
}

#[pymethods]
impl NormalizedPseudoBayesLogistic {
    #[new]
    pub fn new(beta : f64) -> PyResult<Self>{
        Ok(NormalizedPseudoBayesLogistic { beta : beta })
    }

    pub fn call_z0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.z0(y, w, v);
    }

    pub fn call_dz0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.dz0(y, w, v);
    }

    pub fn call_ddz0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.ddz0(y, w, v);
    }
}