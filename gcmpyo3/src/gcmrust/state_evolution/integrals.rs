extern crate optimization;
use std::f64::consts::PI;

use crate::gcmrust::data_models::*;
use crate::gcmrust::channels::*;
use crate::gcmrust::utility::constants::*;

use peroxide::numerical::integral;

pub fn integrate_for_mhat(m : f64, q : f64, v : f64, vstar : f64, channel : &impl base_channel::Channel, data_model : &impl base_partition::Partition) -> f64{
    let mut somme  = 0.0_f64;
    let ys    = [-1.0, 1.0];

    for index in 0..2 {
        let y = ys[index];
        somme = somme + integral::integrate(
            |xi : f64| -> f64 {(- xi.powi(2) / 2.0).exp() / (2.0 * PI).sqrt() * channel.f0(y, q.sqrt() * xi, v) * data_model.dz0(y, m / q.sqrt() * xi, vstar)}, 
            (- INTEGRAL_BOUNDS, INTEGRAL_BOUNDS), integral::Integral::G30K61(GK_PARAMETER)
        );
    }
    
    return somme;

}

pub fn integrate_for_qhat(m : f64, q : f64, v : f64, vstar : f64, channel : &impl base_channel::Channel, data_model : &impl base_partition::Partition) -> f64{
    let mut somme = 0.0_f64;
    let ys    = [-1.0, 1.0];

    for index in 0..2 {
        let y = ys[index];
        somme = somme + integral::integrate(
            |xi : f64| -> f64 {(-xi.powi(2) / 2.0).exp() / (2.0 * PI).sqrt() * channel.f0(y, q.sqrt() * xi, v).powi(2) * data_model.z0(y, m / q.sqrt() * xi, vstar)}, 
            (- INTEGRAL_BOUNDS, INTEGRAL_BOUNDS), integral::Integral::G30K61(GK_PARAMETER)
        );
    }
    return somme;

}

pub fn integrate_for_vhat(m : f64, q : f64, v : f64, vstar : f64, channel : &impl base_channel::Channel, data_model : &impl base_partition::Partition) -> f64{
    let mut somme = 0.0_f64;
    let ys    = [-1.0, 1.0];

    for index in 0..2 {
        let y = ys[index];
        somme = somme + integral::integrate(
            |xi : f64| -> f64 {(-xi.powi(2) / 2.0).exp() / (2.0 * PI).sqrt() * channel.df0(y, q.sqrt() * xi, v) * data_model.z0(y, m / q.sqrt() * xi, vstar)}, 
            (- INTEGRAL_BOUNDS, INTEGRAL_BOUNDS), integral::Integral::G30K61(GK_PARAMETER)
        );
    }
    return somme;
}