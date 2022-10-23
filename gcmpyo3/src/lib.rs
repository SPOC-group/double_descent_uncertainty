use gcmrust::channels;
use gcmrust::data_models;
use pyo3::prelude::*;
use pyo3::wrap_pymodule;

use gcmrust::python::{state_evolution, evidence, utility};

pub mod gcmrust {
    pub mod state_evolution {
        pub mod integrals;
        pub mod state_evolution;
    }

    pub mod data_models {
        pub mod base_partition;
        pub mod base_prior;
     
        pub mod gaussian;

        pub mod logit;
        pub mod probit;
        
        pub mod matching;
        pub mod gcm;
        
    }

    pub mod utility {
        pub mod errors;
        pub mod kappas;
        pub mod evidence;
        pub mod constants;
        pub mod approximation;
    }

    pub mod channels {
        pub mod erm_logistic;
        pub mod pseudo_bayes_logistic;
        pub mod normalized_pseudo_bayes_logistic;
        pub mod base_channel;
        pub mod ridge_regression;
    }

    pub mod python {
        pub mod evidence;
        pub mod state_evolution;
        pub mod utility;
    }

}

// 

#[pyfunction]
fn test() {
    println!("The module is loaded correctly");
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn gcmpyo3(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    m.add_wrapped(wrap_pymodule!(utility::utility))?;
    m.add_wrapped(wrap_pymodule!(evidence::evidence))?;
    m.add_wrapped(wrap_pymodule!(state_evolution::state_evolution))?;

    m.add_class::<channels::pseudo_bayes_logistic::PseudoBayesLogistic>()?;
    m.add_class::<channels::normalized_pseudo_bayes_logistic::NormalizedPseudoBayesLogistic>()?;
    m.add_class::<data_models::logit::Logit>()?;
    m.add_class::<data_models::probit::Probit>()?;

    m.add_function(wrap_pyfunction!(test, m)?)?;

    Ok(())
}
