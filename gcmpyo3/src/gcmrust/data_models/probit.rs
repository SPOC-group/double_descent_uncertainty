use pyo3::prelude::*;

use statrs::function::*;
use peroxide::numerical::integral;
use std::f64::consts::PI;

use crate::gcmrust::data_models::base_partition::Partition;
use crate::gcmrust::utility::constants::*;
use super::base_partition::NormalizedChannel;

#[pyclass(unsendable)]
pub struct Probit {
    pub noise_variance : f64
}

fn probit_likelihood(z : f64) -> f64 {
    0.5 * erf::erfc(- z / 2.0_f64.sqrt())
}

impl Partition for Probit {
    fn z0(&self, y : f64, w  : f64, v : f64) -> f64 {
        // consequence of these lines : REMOVE THE NOISE IN THE DEFINITION OF VSTAR !!!!!!!!
        let noisy_v = v + self.noise_variance;
        return 0.5 * erf::erfc(- (y * w) / (2.0 * noisy_v).sqrt());
    }
    
    fn dz0(&self, y : f64, w  : f64, v : f64) -> f64 {
        // consequence of these lines : REMOVE THE NOISE IN THE DEFINITION OF VSTAR !!!!!!!!
        let noisy_v = v + self.noise_variance;
        return y * (- (w*w) / (2.0 * noisy_v)).exp() / (2.0 * PI * noisy_v).sqrt();
    }

    fn ddz0(&self, y : f64, w : f64, v : f64) -> f64 {
        let noisy_v = v + self.noise_variance;
        return - 2.0 * y * w * (- (w*w) / (2.0 * noisy_v)).exp() / (2.0 * PI * noisy_v).sqrt();
    }
}

impl NormalizedChannel for Probit {
    
}

impl Probit {
    pub fn squared_likelihood_expectation(&self, mean : f64, variance : f64) -> f64 {

        integral::integrate(
            |z : f64| -> f64 { probit_likelihood(( z * variance.sqrt() + mean) / self.noise_variance.sqrt()).powi(2) * (- z*z / 2.0).exp() },
            (-INTEGRAL_BOUNDS, INTEGRAL_BOUNDS),
            integral::Integral::G30K61(GK_PARAMETER)
        ) / (2.0 * PI).sqrt()

    }
}

#[pymethods]
impl Probit {
    #[new]
    pub fn new(noise_variance : f64) -> PyResult<Self>{
        Ok(Probit { noise_variance : noise_variance })
    }

    fn call_z0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.z0(y, w, v);
    }

    fn call_dz0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.dz0(y, w, v);
    }

    fn call_ddz0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.ddz0(y, w, v);
    }
}