// Prior, used for the penalization in ERM for example 

pub trait ParameterPrior {
    // gamma is the ratio student_dim / teacher_dim
    fn get_gamma(&self) -> f64;
    fn get_rho(&self) -> f64;
    fn update_overlaps(&self, mhat : f64, qhat : f64, vhat : f64) -> (f64, f64, f64);
    
    fn update_hatoverlaps_from_integrals(&self, im : f64, iq : f64, iv : f64) -> (f64, f64, f64) {
        return (self.get_gamma().sqrt() * im, iq, -iv);
    }

    fn psi_w(&self, mhat : f64, qhat : f64, vhat : f64) -> f64; 
}

pub trait PseudoBayesPrior {
    fn get_log_prior_strength(&self) -> f64;
}