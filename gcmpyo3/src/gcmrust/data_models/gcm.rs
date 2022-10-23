use crate::gcmrust::data_models::base_prior;
use crate::gcmrust::utility::kappas;

use super::base_prior::ParameterPrior;

pub struct GCMPrior {
    pub kappa1 : f64,
    pub kappastar : f64,
    pub gamma : f64,
    pub lambda : f64,
    pub rho : f64,
    pub teacher_norm : f64
}

pub struct GCMPriorPseudoBayes {
    pub kappa1 : f64,
    pub kappastar : f64,
    pub gamma : f64,
    pub beta_times_lambda : f64,
    pub rho : f64,
    pub teacher_norm : f64
}


pub struct GCMPriorBayesOptimal {
    pub kappa1 : f64,
    pub kappastar : f64,
    pub gamma : f64,
    pub rho : f64,
    pub teacher_norm : f64
}

//////////

impl base_prior::ParameterPrior for GCMPrior {

    fn get_rho(&self) -> f64 {
        return self.rho;
    }
    fn get_gamma(&self) -> f64 {
        return self.gamma;
    }

    fn update_overlaps(&self, mhat : f64, qhat : f64, vhat : f64) -> (f64, f64, f64) {
        let kk1 = self.kappa1.powi(2);
        let kkstar = self.kappastar.powi(2);

        // Refer to the integrals in the GCM paper  (Equations A.34)
        // NOTE : Solve the issue with vhat disappearing from the free energy in the beta -> \infty limit

        let omega_z = |z : f64| -> f64 { kk1 * z + kkstar };
        let phi_t_phi_z = |z : f64| -> f64 { self.teacher_norm * kk1 * z };

        // 1) Compute the integral for v 

        let v_integrand = |z : f64| -> f64 { omega_z(z) / ( self.lambda + vhat * omega_z(z) ) };
        let v : f64 = kappas::marcenko_pastur_integral(&v_integrand, self.get_gamma());

        // 2) Compute the integral for q
        
        let q_integrand = |z : f64 | -> f64 {(qhat * omega_z(z) + mhat.powi(2) * phi_t_phi_z(z)) * omega_z(z) / (self.lambda + vhat * omega_z(z)).powi(2)};
        let q = kappas::marcenko_pastur_integral(&q_integrand, self.get_gamma());

        // 3) Compute the integral for m
        
        let m_integrand = |z : f64| -> f64 { phi_t_phi_z(z) / (self.lambda + vhat * omega_z(z))};
        let m = mhat * self.get_gamma().sqrt() * kappas::marcenko_pastur_integral(&m_integrand, self.get_gamma());
        return (m, q, v);

        /*
        let alpha  = self.gamma;
        let gamma  = 1.0 / self.gamma;
            
        let sigma  = self.kappa1;
        let kk     = self.kappastar * self.kappastar;
        let alphap = ( sigma * (1.0 + alpha.sqrt())).powi(2);
        let alpham = ( sigma * (1.0 - alpha.sqrt())).powi(2);
        if self.lambda == 0.0 {
            let den    = 1.0 + kk * vhat;
            let aux    = (((alphap+kk)*vhat+1.0)*((alpham+kk) * vhat + 1.0)).sqrt();
            let aux2   = (((alphap+kk)*vhat + 1.0) / ((alpham+kk) * vhat + 1.0)).sqrt();
            let mut iv = ((kk*vhat + 1.0) * ((alphap+alpham)*vhat + 2.0) - 2.0 *kk*vhat.powi(2) * (alphap*alpham).sqrt() -2.0 * aux)/(4.0 * alpha*vhat.powi(2)*(kk*vhat+1.0)*sigma.powi(2));
            iv              = iv + 0.0_f64.max(1.0 - gamma)*kk/(1.0 + vhat * kk);
            let i1     = (alphap * vhat*(-3.0 * den+aux)+4.0 * den * (-den+aux)+alpham*vhat*(-2.0 * alphap * vhat - 3.0 * den + aux))/(4.0 * alpha * vhat.powi(3) * sigma.powi(2)*aux);
            let i2     = (alphap * vhat+alpham*vhat*(1.0 - 2.0 * aux2) + 2.0 * den * (1.0 - aux2))/(4.0 * alpha * vhat.powi(2) * aux * sigma.powi(2));
            let i3     = (2.0 * vhat * alphap*alpham+(alphap+alpham) * den- 2.0 * (alphap*alpham).sqrt() * aux)/(4.0 * alpha * den.powi(2) * sigma.powi(2) * aux);
            let mut iq = (qhat + mhat.powi(2)) * i1 + (2.0*qhat+mhat.powi(2)) * kk * i2 + qhat * kk.powi(2) * i3;
            iq              = iq + 0.0_f64.max(1.0-gamma)*qhat*(kk / den).powi(2);
            let im     = ((alpham + alphap+2.0*kk)*vhat+2.0 - 2.0 * aux)/(4.0*alpha*(vhat/sigma).powi(2));
            let v = iv;
            let m = mhat * (self.gamma).sqrt() * im;
            let q = iq;
            return (m, q, v);
        }

        else {
            let den    = self.lambda+kk*vhat;
            let aux    = ( ((alphap+kk)*vhat+self.lambda) * ((alpham+kk)*vhat+self.lambda)).sqrt();
            let aux2   = ( ((alphap+kk)*vhat+self.lambda) /( (alpham+kk)*vhat+self.lambda)).sqrt();
            let mut iv = ((kk*vhat+self.lambda)*((alphap+alpham)*vhat+2.0 * self.lambda)-2.0 * kk*vhat.powi(2)*(alphap*alpham).sqrt()-2.0 * self.lambda*aux)/(4.0 * alpha*vhat.powi(2)*(kk*vhat+self.lambda)*sigma.powi(2));
            iv              = iv + f64::max(0.0, 1.0-gamma)*kk / (self.lambda + vhat * kk);
            let i1     = (alphap*vhat*(-3.0*den+aux)+4.0*den*(-den+aux)+alpham*vhat*(-2.0*alphap*vhat-3.0*den+aux))/(4.0*alpha*vhat.powi(3)*sigma.powi(2)*aux);
            let i2     = (alphap*vhat+alpham*vhat*(1.0-2.0*aux2)+2.0*den*(1.0-aux2))/(4.0*alpha*vhat.powi(2)*aux*sigma.powi(2));
            let i3     = (2.0 * vhat * alphap * alpham+(alphap + alpham) * den - 2.0 * (alphap*alpham).sqrt()* aux) / (4.0 * alpha * den.powi(2) * sigma.powi(2)* aux) ;
            let mut iq = (qhat+mhat.powi(2))*i1+(2.0*qhat+mhat.powi(2))*kk*i2+qhat*kk.powi(2)*i3;
            iq              = iq + f64::max(0.0, 1.0 - gamma)*qhat*kk.powi(2)/den.powi(2);
            let im     = ((alpham+alphap+2.0*kk)*vhat+2.0*self.lambda-2.0*aux)/(4.0*alpha*vhat.powi(2)*sigma.powi(2));


            let v = iv;
            let m = mhat * (self.gamma).sqrt() * im;
            let q = iq;
            return (m, q, v);
        } 
        */
    }

    fn psi_w(&self, mhat : f64, qhat : f64, vhat : f64) -> f64 {
        let (kk1, kkstar) = (self.kappa1 * self.kappa1, self.kappastar * self.kappastar);
        let to_integrate_1 = |x : f64| -> f64 {(self.lambda + vhat * (kk1 * x + kkstar)).ln()};
        let to_integrate_2 = |x : f64| -> f64 { (mhat * kk1 * x + qhat * (kk1 * x + kkstar)) / (self.lambda + vhat * (kk1 * x + kkstar))} ;
        return - 0.5 * kappas::marcenko_pastur_integral(&to_integrate_1, self.gamma) + 0.5 * kappas::marcenko_pastur_integral(&to_integrate_2, self.gamma);
    }

}

// 

impl base_prior::ParameterPrior for GCMPriorPseudoBayes {

    fn get_rho(&self) -> f64 {
        return self.rho;
    }
    fn get_gamma(&self) -> f64 {
        return self.gamma;
    }

    fn update_overlaps(&self, mhat : f64, qhat : f64, vhat : f64) -> (f64, f64, f64) {
        /* 
        let alpha      = self.gamma;
        let gamma      = 1.0 / self.gamma;
                
        let sigma      = self.kappa1;
        let kk         = self.kappastar * self.kappastar;
        let alphap     = ( sigma * (1.0 + alpha.sqrt())).powi(2);
        let alpham     = ( sigma * (1.0 - alpha.sqrt())).powi(2);
        if  self.beta_times_lambda == 0.0 {
            let den    = 1.0 + kk * vhat;
            let aux    = (((alphap+kk)*vhat+1.0)*((alpham+kk) * vhat + 1.0)).sqrt();
            let aux2   = (((alphap+kk)*vhat + 1.0) / ((alpham+kk) * vhat + 1.0)).sqrt();
            let mut iv = ((kk*vhat + 1.0) * ((alphap+alpham)*vhat + 2.0) - 2.0 *kk*vhat.powi(2) * (alphap*alpham).sqrt() -2.0 * aux)/(4.0 * alpha*vhat.powi(2)*(kk*vhat+1.0)*sigma.powi(2));
            iv              = iv + 0.0_f64.max(1.0 - gamma)*kk/(1.0 + vhat * kk);
            let i1     = (alphap * vhat*(-3.0 * den+aux)+4.0 * den * (-den+aux)+alpham*vhat*(-2.0 * alphap * vhat - 3.0 * den + aux))/(4.0 * alpha * vhat.powi(3) * sigma.powi(2)*aux);
            let i2     = (alphap * vhat+alpham*vhat*(1.0 - 2.0 * aux2) + 2.0 * den * (1.0 - aux2))/(4.0 * alpha * vhat.powi(2) * aux * sigma.powi(2));
            let i3     = (2.0 * vhat * alphap*alpham+(alphap+alpham) * den- 2.0 * (alphap*alpham).sqrt() * aux)/(4.0 * alpha * den.powi(2) * sigma.powi(2) * aux);
            let mut iq = (qhat + mhat.powi(2)) * i1 + (2.0*qhat+mhat.powi(2)) * kk * i2 + qhat * kk.powi(2) * i3;
            iq              = iq + 0.0_f64.max(1.0-gamma)*qhat*(kk / den).powi(2);
            let im     = ((alpham + alphap+2.0*kk)*vhat+2.0 - 2.0 * aux)/(4.0*alpha*(vhat/sigma).powi(2));
            let v      = iv;
            let m      = mhat * (self.gamma).sqrt() * im;
            let q      = iq;
            return (m, q, v);
        }

        else {
            let den    =  self.beta_times_lambda + kk * vhat;
            let aux    = ( ((alphap+kk)*vhat+ self.beta_times_lambda) * ((alpham+kk)*vhat+ self.beta_times_lambda)).sqrt();
            let aux2   = ( ((alphap+kk)*vhat+ self.beta_times_lambda) /( (alpham+kk)*vhat+ self.beta_times_lambda)).sqrt();
            let mut iv = ((kk*vhat+ self.beta_times_lambda)*((alphap+alpham)*vhat+2.0 *  self.beta_times_lambda)-2.0 * kk*vhat.powi(2)*(alphap*alpham).sqrt()-2.0 *  self.beta_times_lambda*aux)/(4.0 * alpha*vhat.powi(2)*(kk*vhat+ self.beta_times_lambda)*sigma.powi(2));
            iv              = iv + f64::max(0.0, 1.0 - gamma ) * kk / ( self.beta_times_lambda + vhat * kk );
            let i1     = (alphap*vhat*(-3.0*den+aux)+4.0*den*(-den+aux)+alpham*vhat*(-2.0*alphap*vhat-3.0*den+aux))/(4.0*alpha*vhat.powi(3)*sigma.powi(2)*aux);
            let i2     = (alphap*vhat+alpham*vhat*(1.0-2.0*aux2)+2.0*den*(1.0-aux2))/(4.0*alpha*vhat.powi(2)*aux*sigma.powi(2));
            let i3     = (2.0 * vhat * alphap * alpham+(alphap + alpham) * den - 2.0 * (alphap*alpham).sqrt()* aux) / (4.0 * alpha * den.powi(2) * sigma.powi(2)* aux) ;
            let mut iq = (qhat+mhat.powi(2))*i1+(2.0*qhat+mhat.powi(2))*kk*i2+qhat*kk.powi(2)*i3;
            iq              = iq + f64::max(0.0, 1.0 - gamma)*qhat*kk.powi(2)/den.powi(2);
            let im     = ((alpham+alphap+2.0*kk)*vhat+2.0* self.beta_times_lambda-2.0*aux)/(4.0*alpha*vhat.powi(2)*sigma.powi(2));


            let v = iv;
            let m = mhat * (self.gamma).sqrt() * im;
            let q = iq;
            return (m, q, v);
        }  
        */

        // This version seems to work

        let kk1 = self.kappa1.powi(2);
        let kkstar = self.kappastar.powi(2);

        // Refer to the integrals in the GCM paper  (Equations A.34)

        let omega_z = |z : f64| -> f64 { kk1 * z + kkstar };
        let phi_t_phi_z = |z : f64| -> f64 { self.teacher_norm * kk1 * z };

        // 1) Compute the integral for v 

        let v_integrand = |z : f64| -> f64 { omega_z(z) / ( self.beta_times_lambda + vhat * omega_z(z) ) };
        let v : f64 = kappas::marcenko_pastur_integral(&v_integrand, self.get_gamma());

        // 2) Compute the integral for q
        
        let q_integrand = |z : f64 | -> f64 {(qhat * omega_z(z) + mhat.powi(2) * phi_t_phi_z(z)) * omega_z(z) / (self.beta_times_lambda + vhat * omega_z(z)).powi(2)};
        let q = kappas::marcenko_pastur_integral(&q_integrand, self.get_gamma());

        // 3) Compute the integral for m
        
        let m_integrand = |z : f64| -> f64 {phi_t_phi_z(z) / (self.beta_times_lambda + vhat * omega_z(z))};
        let m = mhat * self.get_gamma().sqrt() * kappas::marcenko_pastur_integral(&m_integrand, self.get_gamma());
        return (m, q, v);
    }
    
    fn psi_w(&self, mhat : f64, qhat : f64, vhat : f64) -> f64 {
        let (kk1, kkstar) = (self.kappa1 * self.kappa1, self.kappastar * self.kappastar);

        let omega_z = |z : f64| -> f64 { kk1 * z + kkstar };
        let phi_t_phi_z = |z : f64| -> f64 { self.teacher_norm * kk1 * z };

        let to_integrate_1 = |x : f64| -> f64 { (self.beta_times_lambda + vhat * omega_z(x)).ln() };
        let to_integrate_2 = |x : f64| -> f64 { (mhat.powi(2) * phi_t_phi_z(x) + qhat * omega_z(x)) / (self.beta_times_lambda + vhat * omega_z(x)) };

        return - 0.5 * kappas::marcenko_pastur_integral(&to_integrate_1, self.gamma) + 0.5 * kappas::marcenko_pastur_integral(&to_integrate_2, self.gamma);
    }

}

impl base_prior::PseudoBayesPrior for GCMPriorPseudoBayes {
    fn get_log_prior_strength(&self) -> f64 {
        return (self.beta_times_lambda).ln();
    }
}

//

impl base_prior::ParameterPrior for GCMPriorBayesOptimal {
    fn get_gamma(&self) -> f64 {
        return self.gamma;
    }

    fn get_rho(&self) -> f64 {
        return self.rho;
    }

    fn update_overlaps(&self, mhat : f64, qhat : f64, vhat : f64) -> (f64, f64, f64) {
        let kk1 = self.kappa1.powi(2);
        let kkstar = self.kappastar.powi(2);
        let q = self.get_gamma() * qhat * kappas::marcenko_pastur_integral(&|z : f64| -> f64 {(kk1 * z * self.teacher_norm / (kk1 * z + kkstar)).powi(2)  / (1.0 + qhat * (kk1 * z * self.teacher_norm / (kk1 * z + kkstar)))}, self.get_gamma());

        return (q, q, self.get_rho() - q);
    }

    fn psi_w(&self, mhat : f64, qhat : f64, vhat : f64) -> f64 {
        let kk1 = self.kappa1.powi(2);
        let kkstar = self.kappastar.powi(2);
        let to_integrate = |z : f64| -> f64 { qhat * (kk1 * z) / (kkstar + kk1 * z) - (qhat + (kk1 * z + kkstar) / (kk1 * z)).ln()};
        return 0.5 * self.get_gamma() * kappas::marcenko_pastur_integral_without_zero(&to_integrate, self.get_gamma());
    }

    fn update_hatoverlaps_from_integrals(&self, im : f64, iq : f64, iv : f64) -> (f64, f64, f64) {
        return (self.gamma * iq, self.gamma * iq, self.gamma * iq);
    }
}

impl base_prior::PseudoBayesPrior for GCMPriorBayesOptimal {
    fn get_log_prior_strength(&self) -> f64 {
        // I may be doing a mistake : the normalization might be already included in psi_w, cf the overleaf [scratch] overparam. uncetainty
        // In any case, the evidence does not depend on any hyperparameter so it should not matter much

        // log of det is trace log, here just sum the non-zero eigenvalues
        let kk1 = self.kappa1.powi(2);
        let kkstar = self.kappastar.powi(2);
        // Need to not include the 0, because the log witll not be determined
        return kappas::marcenko_pastur_integral_without_zero(&|z : f64| -> f64 { z.ln() + kk1.ln() - 2.0 * (kk1 * z + kkstar).ln() }, self.get_gamma());
    }
}