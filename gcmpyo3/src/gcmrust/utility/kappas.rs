use std::f64::consts::PI;
use peroxide::numerical::*;
use crate::gcmrust::utility::constants::*;

pub fn get_kappas_from_activation(activation : &String) -> (f64, f64) {
    return match activation.as_str() {
        "erf" => (2.0 / (3.0 * PI).sqrt(), 0.200364),
        "relu" => (0.5, ((PI - 2.0)/(4.0 * PI)).sqrt()),
        _ => panic!()
    };
}

pub fn get_additional_noise_variance_from_kappas(kappa1 : f64, kappastar : f64, gamma : f64) -> f64 {
    let kk1     = kappa1.powi(2);
    let kkstar  = kappastar.powi(2);

    let lambda_minus = (1.0 - gamma.sqrt()).powi(2);
    let lambda_plus       = (1.0 + gamma.sqrt()).powi(2);

    let integrand = |lambda : f64| -> f64 {((lambda_plus - lambda ) * (lambda - lambda_minus)).sqrt() / (kkstar + kk1 * lambda)};
    
    return 1.0 - kk1 * integral::integrate(integrand, (lambda_minus, lambda_plus), integral::Integral::G30K61(GK_PARAMETER)) / (2.0 * PI);
}

pub fn marcenko_pastur_integral(f : &dyn Fn(f64) -> f64, gamma : f64) -> f64 {
    let (lambda_minus, lambda_plus) : (f64, f64) = ((1.0 - gamma.sqrt()).powi(2), (1.0 + gamma.sqrt()).powi(2));
    let to_integrate = |x : f64| -> f64 {f(x) * ((lambda_plus - x) * (x - lambda_minus)).sqrt() / (2.0 * PI * gamma * x)};
    let integral : f64 = integral::integrate(to_integrate, (lambda_minus, lambda_plus), integral::Integral::G30K61(GK_PARAMETER));
    if gamma > 1.0 {
        return integral + (1.0 - 1.0 / gamma) * f(0.0);
    }
    return integral
}

pub fn marcenko_pastur_integral_without_zero(f : &dyn Fn(f64) -> f64, gamma : f64) -> f64 {
    let (lambda_minus, lambda_plus) : (f64, f64) = ((1.0 - gamma.sqrt()).powi(2), (1.0 + gamma.sqrt()).powi(2));
    let to_integrate = |x : f64| -> f64 {f(x) * ((lambda_plus - x) * (x - lambda_minus)).sqrt() / (2.0 * PI * gamma * x)};
    let integral : f64 = integral::integrate(to_integrate, (lambda_minus, lambda_plus), integral::Integral::G30K61(GK_PARAMETER));
    return integral
}